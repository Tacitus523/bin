#!/usr/bin/env python3

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from datetime import datetime

# Import Keras Tuner
try:
    import keras_tuner as kt
    import tensorflow as tf
except ImportError as e:
    print("Error: Keras Tuner not found. Please install with: pip install keras-tuner")
    sys.exit(1)

ENERGY_UNIT = "eV"
FORCE_UNIT = r"$\frac{\mathrm{eV}}{\mathrm{\AA}}$"

SCORE_LABEL = f"Force RMSE ({FORCE_UNIT})"
COLORMAP_NAME_NUMERICAL = 'YlOrRd'
COLORMAP_NAME_CATEGORICAL = 'tab10'
SINGLE_FIG_SIZE=(12, 6)
MULTIPLOT_FIG_SIZE=(5, 4)
DPI = 150

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze Keras Tuner trials and create visualizations"
    )
    parser.add_argument(
        "-t", "--trial_dir",
        type=str,
        default="trials",
        help="Path to Keras Tuner trial directory"
    )
    parser.add_argument(
        "-p", "--project-names",
        dest="project_names",
        nargs="+",
        type=str,
        default=None,
        help="Keras Tuner project name(s) (default: None)"
    )
    parser.add_argument(
        "-n", "--n-best",
        type=int,
        default=12,
        help="Number of best trials to analyze (default: 12)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="trial_analysis",
        help="Output directory for CSV and plots (default: trial_analysis)"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="score",
        help="Metric to optimize for (default: score)"
    )
    parser.add_argument(
        "--minimize",
        action="store_true",
        default=True,
        help="Whether to minimize the metric (default: True)"
    )
    parser.add_argument(
        "--maximize",
        action="store_false",
        dest="minimize",
        help="Whether to maximize the metric"
    )
    parser.add_argument(
        "--grid_search",
        action="store_true",
        default=True,
        help="Whether the tuner used grid search"
    )
    parser.add_argument(
        "--hyperband",
        action="store_false",
        dest="grid_search",
        help="Whether the tuner used Hyperband"
    )
    args = parser.parse_args()
    args.trial_dir = os.path.abspath(args.trial_dir)
    if not os.path.exists(args.trial_dir):
        raise FileNotFoundError(f"Trial directory not found: {args.trial_dir}")
    return args

class DummyTuner(kt.RandomSearch):
    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters
        x = hp.Float("x", min_value=-1.0, max_value=1.0)
        return x

def calculate_trial_duration(trial_id: str, trial_start_times: Dict[str, datetime], trial_folder: str) -> Optional[float]:
    """Calculate approximate trial duration from start times."""
    if trial_id not in trial_start_times:
        return None
    
    # Get current trial start time
    current_start = trial_start_times[trial_id]
    
    # Try to find next trial to estimate duration
    next_trial_id = trial_id[:-1] + str(int(trial_id[-1]) + 1)  # Increment last character
    
    if next_trial_id in trial_start_times:
        next_start = trial_start_times[next_trial_id]
        duration_seconds = (next_start - current_start).total_seconds()
        return duration_seconds
    else:
        return np.nan
        # For the last trial, we can't estimate duration this way, estimate from file modification time
        trial_file = os.path.join(trial_folder, "trial.json")
        if os.path.exists(trial_file):
            mod_time = os.path.getmtime(trial_file)
            duration_seconds = mod_time - current_start.timestamp()
            return duration_seconds if duration_seconds > 0 else None
        else:
            raise FileNotFoundError(f"Trial file not found: {trial_file}")

def load_tuner_and_extract_data(directory: str, project_name: str) -> pd.DataFrame:
    """Load trial data using Keras Tuner API."""
    directory_path = Path(directory)
    
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    if not os.path.exists(os.path.join(directory, project_name)):
        raise FileNotFoundError(f"Project name directory not found: {project_name}")
    
    try:
        tuner = DummyTuner(
            directory=directory,
            project_name=project_name,
            overwrite=False  # Don't overwrite existing trials
        )
        
        if len(tuner.oracle.trials) == 0:
            raise ValueError(f"No trials found in the specified directory {directory} and project name {project_name}.")
        
    except Exception as e:
        print(f"Error creating tuner: {e}")
        raise
    
    # Extract timing information from oracle.json
    trial_start_times: List[datetime] = tuner.oracle._display.trial_start
    
    # Extract trial data using tuner API
    trial_data = []
    
    for trial_id, trial in tuner.oracle.trials.items():
        # Extract basic trial information
        trial_data_point = {
            'project_name': project_name,
            'trial_id': trial_id,
            'status': trial.status,
            'score': trial.score
        }
        
        # Extract hyperparameters
        for hp_name, hp_value in trial.hyperparameters.values.items():
            if hp_name.startswith('tuner/'):
                continue
            trial_data_point[f"param_{hp_name}"] = hp_value
        
        # Extract all metrics
        for metric_name, metric_observations in trial.metrics.metrics.items():
            values_at_step = metric_observations.get_best_value() # Just one value in history anyway
            if "time" in metric_name.lower():
                trial_data_point[metric_name] = values_at_step
            else:
                trial_data_point[f"metric_{metric_name}"] = values_at_step
        
        # Extract timing information, used if no history timing available
        if trial_id in trial_start_times:
            trial_folder = tuner.get_trial_dir(trial_id)
            duration = calculate_trial_duration(trial_id, trial_start_times, trial_folder)
            trial_data_point['duration'] = duration // 60 # Convert to minutes
        
        trial_data.append(trial_data_point)

    if len(trial_data) == 0:
        print(f"Warning: No trial data extracted for project {project_name} in directory {directory}")

    return pd.DataFrame(trial_data)

def get_best_trials(trials_df: pd.DataFrame,
                   metric: str, 
                   n_best: int, 
                   minimize: bool = True) -> pd.DataFrame:
    """Get the best n trials based on the specified metric."""
    
    # Check if metric exists
    metric_col = None
    if metric in trials_df.columns:
        metric_col = metric
    elif f"metric_{metric}" in trials_df.columns:
        metric_col = f"metric_{metric}"
    elif "score" in trials_df.columns:
        metric_col = "score"
        print(f"Warning: Metric '{metric}' not found, using 'score' instead")
    else:
        raise ValueError(f"Metric '{metric}' not found in trial data. Available columns: {list(trials_df.columns)}")
    
    # Sort trials by metric
    df_sorted = trials_df.sort_values(metric_col, ascending=minimize)
    
    # Get best n trials
    best_trials = df_sorted.head(n_best).copy()
    best_trials['rank'] = range(1, len(best_trials) + 1) # Add rank column
    best_trials = best_trials.sort_values(["project_name", "trial_id"]).reset_index(drop=True)
    
    return best_trials

def clean_trials(trials_df: pd.DataFrame) -> pd.DataFrame:
    """Clean the trials DataFrame."""

    layer_columns = [col for col in trials_df.columns if '_n_layers' in col]

    # *_layer groups contain a value for higher layer neurons, even if they are not used
    def layer_cleanup(data_point: pd.Series) -> pd.Series:
        for layer_col in layer_columns:
            prefix = layer_col.replace('_n_layers', '')
            neuron_columns = [col for col in trials_df.columns if col.startswith(prefix) and "_neurons_" in col]
            if layer_col in data_point:
                n_layers = data_point[layer_col]
                if pd.isna(n_layers):
                    continue
                for neuron_col in neuron_columns:
                    layer_index = int(neuron_col.split('_neurons_')[-1])
                    if layer_index >= n_layers:
                        data_point[neuron_col] = np.nan
        return data_point
    
    trials_df = trials_df.copy().apply(layer_cleanup, axis=1)
    
    # Keep neuron columns as int dtype (convert NaN to nullable Int64)
    neuron_columns = [col for col in trials_df.columns if "_neurons_" in col]
    for col in neuron_columns:
        trials_df[col] = trials_df[col].astype('Int64')
    
    return trials_df

def categorize_parameters(best_trials_df: pd.DataFrame, all_trials_df: pd.DataFrame) -> pd.DataFrame:
    """Convert parameter columns to categorical with all possible values from all trials."""
    best_trials_df = best_trials_df.copy()
    
    # Find all parameter columns
    param_cols = [col for col in best_trials_df.columns if col.startswith("param_")]
    
    for col in param_cols:
        # Get all unique values from all trials (the complete set of categories)
        all_categories = all_trials_df[col].dropna().unique()
        
        # Convert to categorical with all possible categories
        best_trials_df[col] = pd.Categorical(
            best_trials_df[col], 
            categories=sorted(all_categories),
            ordered=True,
        )
    
    return best_trials_df

def save_trials_csv(trials_df: pd.DataFrame, output_path: str) -> None:
    """Save trials data to CSV file."""
    trials_df.to_csv(output_path, index=False)
    print(f"Saved best trials data to: {output_path}")

def create_trial_identifiers(top_rank_df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Create trial identifiers combining project name and trial ID."""
    identifiers = []
    for _, row in top_rank_df.iterrows():
        clean_project_name = row['project_name'].replace('_', ' ').title()
        clean_trial_id = row['trial_id']#f"Trial {row['trial_id']}"
        if verbose:
            identifier = f"{clean_project_name}\n{clean_trial_id}"
        else:
            identifier = clean_trial_id
        identifiers.append(identifier)
    
    top_rank_df = top_rank_df.copy()
    top_rank_df['identifier'] = identifiers
    return top_rank_df

def get_colors(max_value: float, values: List[float]) -> List[Tuple[float, float, float, float]]:
    """Generate a list of colors from the colormap based on max value."""
    metric_max = max_value
    a = 0
    norm = plt.Normalize(vmin=a*metric_max, vmax=metric_max)
    cmap = sns.color_palette(COLORMAP_NAME_NUMERICAL, as_cmap=True)
    colors = [cmap(norm(value)) for value in values]
    return colors

def plot_grid_search_analysis(trials_df: pd.DataFrame, 
                                metric_col: str,
                                output_dir: str,
                                target_project_name: Optional[str] = None ) -> None:
    """Create bar plots for hyperparameter analysis."""

    # Normalize metric values for colormap
    metric_max = trials_df[metric_col].max()

    if target_project_name is not None:
        trials_df = trials_df[trials_df['project_name'] == target_project_name]

    trials_df = trials_df.dropna(how='all', axis=1)
    trials_df = trials_df.sort_values(["project_name", "trial_id"])

    project_names = trials_df["project_name"].unique().tolist()
    
    # Create subplots
    n_projects = len(project_names)
    n_cols = min(3, n_projects)
    n_rows = (n_projects + n_cols - 1) // n_cols
    if n_projects == 1:
        fig, axes = plt.subplots(1, 1, figsize=SINGLE_FIG_SIZE)
    else:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(MULTIPLOT_FIG_SIZE[0]*n_cols, MULTIPLOT_FIG_SIZE[1]*n_rows))
    if n_projects == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if n_projects > 1 else [axes]
    else:
        axes = axes.flatten()
    
    for i, project_name in enumerate(project_names):
        ax = axes[i] if i < len(axes) else None
        if ax is None:
            continue

        project_df = trials_df[trials_df["project_name"] == project_name]
        clean_project_name = project_name.replace('_', ' ').title()
        #clean_metric_name = metric_col.replace('metric_', '').replace('_', ' ').title()
        
        # Create colors based on metric values
        colors = get_colors(metric_max, project_df[metric_col])
        sns.barplot(x="trial_id", y=metric_col, data=project_df, ax=ax, palette=colors, hue="trial_id", legend=False)
        ax.set_title(clean_project_name if target_project_name is None else None)
        ax.set_xlabel('Trial ID')
        ax.set_ylabel(SCORE_LABEL)
        ax.set_ylim(0, metric_max*1.1)
    
    # Hide unused subplots
    for i in range(n_projects, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    filename = f"hyperparameter_analysis_{project_name}.png" if target_project_name else "hyperparameter_analysis.png"
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path, dpi=DPI, bbox_inches='tight')
    plt.close()

def plot_trial_ranking(trials_df: pd.DataFrame, 
                      metric_col: str,
                      output_dir: str,
                      target_project_name: Optional[str] = None ) -> None:
    """Create a bar plot showing trial rankings."""
    
    max_metric = trials_df[metric_col].max()

    if target_project_name is not None:
        trials_df = trials_df[trials_df['project_name'] == target_project_name]
    project_names = trials_df["project_name"].unique().tolist()
    n_projects = len(project_names)
    verbose = target_project_name is None and n_projects > 1 # Only print project names if not analyzing a specific one


    top_rank_df = trials_df.sort_values(metric_col).head(10).reset_index(drop=True)
    
    plt.figure(figsize=SINGLE_FIG_SIZE)
    
    # Create bar plot with colors based on metric values
    colors = get_colors(max_metric, top_rank_df[metric_col])

    top_rank_df = create_trial_identifiers(top_rank_df, verbose=verbose)
    
    ax = sns.barplot(x='identifier', y=metric_col, data=top_rank_df, 
                     palette=colors, hue='identifier', legend=False)
    
    plt.xlabel('Trial ID')
    #plt.ylabel(metric_col.replace('metric_', '').replace('_', ' ').title())
    plt.ylabel(SCORE_LABEL)
    #plt.title(f'Best {len(top_rank_df)} Trials - {metric_col}')
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(top_rank_df.iterrows()):
        value = row[metric_col]
        ax.text(i, value, f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    filename = f"trial_ranking_{target_project_name}.png" if target_project_name else "trial_ranking.png"
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path, dpi=DPI, bbox_inches='tight')
    plt.close()

def plot_duration_analysis(trials_df: pd.DataFrame, 
                          metric_col: str,
                          output_dir: str,
                          target_project_name: Optional[str] = None,
                          do_sort: bool = False) -> None:
    """Create plots analyzing trial duration vs performance."""

    max_metric = trials_df[metric_col].max()

    trials_df = trials_df.copy()
    if do_sort:
        trials_df = trials_df.sort_values([metric_col, "project_name", "trial_id"], ascending=[True, True, True])
    else:
        trials_df = trials_df.sort_values(["project_name", "trial_id"], ascending=[True, True])
    
    possible_timing_keys = ['total_training_time', 'duration', 'elapsed_time', 'elapsed_process_time', 'total_process_time']
    if not any(key in trials_df.columns for key in possible_timing_keys):
        print("Warning: No timing information found in trials data")
        return

    real_time_key = None
    possible_real_time_keys = ['total_training_time', 'duration', 'elapsed_time']
    for key in possible_real_time_keys:
        if key in trials_df.columns:
            real_time_key = key
            trials_df = trials_df.dropna(subset=[real_time_key])
            break
    if real_time_key is None:
        print("Warning: No real time information found in trials data")
        return

    process_time_key = None
    possible_process_time_keys = ['elapsed_process_time', 'total_process_time'] # total_process_time seems bugged, way too high
    for key in possible_process_time_keys:
        if key in trials_df.columns:
            process_time_key = key
            trials_df = trials_df.dropna(subset=[process_time_key])
            break
    if process_time_key is None:
        print("Warning: No process time information found in trials data")
    
    if len(trials_df) == 0:
        print("Warning: No valid duration data found")
        return

    project_names = trials_df["project_name"].unique().tolist()
    n_projects = len(project_names)

    verbose = target_project_name is None and n_projects > 1 # Only print project names if not analyzing a specific one

    if target_project_name is not None:
        trials_df = trials_df[trials_df['project_name'] == target_project_name]
    
    # Create subplots
    n_cols = min(3, n_projects)
    n_rows = (n_projects + n_cols - 1) // n_cols
    
    if n_projects == 1:
        fig, axes = plt.subplots(1, 1, figsize=SINGLE_FIG_SIZE)
    else:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(MULTIPLOT_FIG_SIZE[0]*n_cols, MULTIPLOT_FIG_SIZE[1]*n_rows))
    if n_projects == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if n_projects > 1 else [axes]
    else:
        axes = axes.flatten()
    
    for i, project_name in enumerate(project_names):
        project_df = trials_df[trials_df["project_name"] == project_name]
        clean_project_name = project_name.replace('_', ' ').title()
        ax = axes[i] if i < len(axes) else None
        if ax is None:
            continue

        project_df = create_trial_identifiers(project_df, verbose=verbose)

        colors = get_colors(max_metric, project_df[metric_col])
        sns.barplot(x='identifier', y=real_time_key, data=project_df, 
            palette=colors, hue='identifier', legend=False, ax=ax)
        ax.set_title(clean_project_name if verbose else None)
        ax.set_xlabel('Trial ID')
        ax.set_ylabel('Duration (minutes)')
        # ax.set_xticks(ax.get_xticks())
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')

    # Hide unused subplots
    for i in range(n_projects, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    filename = f"duration_analysis_{target_project_name}.png" if target_project_name else "duration_analysis.png"
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path, dpi=DPI, bbox_inches='tight')
    plt.close()

    if target_project_name is None:
        print(f"\nDuration Statistics:")
        print(f"Mean duration: {trials_df[real_time_key].mean():.1f} minutes")
        print(f"Median duration: {trials_df[real_time_key].median():.1f} minutes")
        print(f"Min duration: {trials_df[real_time_key].min():.1f} minutes")
        print(f"Max duration: {trials_df[real_time_key].max():.1f} minutes")

def plot_hyperband_analysis(trials_df: pd.DataFrame, 
                           metric_col: str,
                           output_dir: str,
                           target_param_name: Optional[str] = None
                           ) -> None:
    """Create Hyperband-specific analysis plots. These contain multiple paramters per trial."""

    do_legend = target_param_name is not None

    hyperparam_column_names = [col for col in trials_df.columns if col.startswith('param_')] if target_param_name is None else [target_param_name]

    # Create subplots
    n_params = len(hyperparam_column_names)
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols
    
    if n_params == 1:
        fig, axes = plt.subplots(1, 1, figsize=SINGLE_FIG_SIZE)
    else:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(MULTIPLOT_FIG_SIZE[0]*n_cols, MULTIPLOT_FIG_SIZE[1]*n_rows))
    if n_params == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if n_params > 1 else [axes]
    else:
        axes = axes.flatten()

    for i, column_name in enumerate(hyperparam_column_names):
        ax = axes[i] if i < len(axes) else None
        if ax is None:
            continue
        
        param_df = trials_df.copy().dropna(subset=[column_name], axis=0)
        param_df = param_df.sort_values(column_name)
        clean_param_name = column_name.replace('param_', '').replace('_', ' ').title()
        
        # Get all categories including missing ones
        all_categories = trials_df[column_name].cat.categories if isinstance(trials_df[column_name].dtype, pd.CategoricalDtype) else sorted(trials_df[column_name].dropna().unique())
               
        # Create parameter ID mapping for all categories
        param_id_mapping = {cat: idx for idx, cat in enumerate(all_categories)}
        column_id_mapping = {idx: cat for idx, cat in enumerate(all_categories)}
        param_df["param_id"] = param_df[column_name].map(param_id_mapping)
        
        # Create order for x-axis to include all categories
        x_order = list(range(len(all_categories)))

        do_label_xticks=False
        if all([len(str(category))<10 for category in all_categories]):
            do_label_xticks = True

        # Box plot with swarm plot overlay for all parameter types
        # sns.boxplot(x=plot_column, y=metric_col, data=param_df, 
        #     palette=COLORMAP_NAME_CATEGORICAL, hue=column_name, ax=ax, legend=do_legend)
        sns.swarmplot(x="param_id", y=metric_col, data=param_df, hue="param_id", order=x_order,
            palette=COLORMAP_NAME_CATEGORICAL, marker="x", linewidth=2, size=8, ax=ax, legend=do_legend)
        ax.set_title(clean_param_name if not do_legend else None)
        ax.set_xlabel("Parameter ID")
        ax.set_ylabel(SCORE_LABEL)
        if do_label_xticks:
            ax.set_xlabel(clean_param_name)
            ax.set_xticks(x_order)
            ax.set_xticklabels([column_id_mapping[idx] for idx in x_order])

        if not do_legend:
            continue

        if do_label_xticks:
            continue

        # Modify Legend
        handles, labels = ax.get_legend_handles_labels()
        # value_counts = param_df[column_name].value_counts()
        # value_counts.index = value_counts.index.astype(str)
        # is_present = value_counts[value_counts > 0].index

        new_handles, new_labels = [], []
        
        # First, add present labels in order of appearance
        for handle, label in zip(handles, labels):
            #if label in is_present:
                #new_label = f"{label}: n={value_counts.loc[label]}"
                new_label = f"{label}: {column_id_mapping[int(label)]}"
                handle.set_alpha(1)
                new_handles.append(handle)
                new_labels.append(new_label)

        # # Then add absent labels
        # for handle, label in zip(handles, labels):
        #     if label not in is_present:
        #         new_label = f"{label}: n={value_counts.loc[label]}"
        #         new_handles.append(handle)
        #         new_labels.append(new_label)

        ax.legend(new_handles, new_labels, title=clean_param_name, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Hide unused subplots
    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)

    if not do_legend:
        plt.tight_layout()
    # Save plot
    filename = f"hyperparameter_analysis_{target_param_name.replace('param_', '')}.png" if target_param_name else "hyperparameter_analysis.png"
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path, dpi=DPI, bbox_inches='tight')
    plt.close()

    # # Obsolete Barplot
    # for i, column_name in enumerate(hyperparam_column_names):
    #     ax = axes[i] if i < len(axes) else None
    #     if ax is None:
    #         continue
        
    #     clean_param_name = column_name.replace('param_', '').replace('_', ' ').title()
    #     trials_df["param_id"] = pd.factorize(trials_df[column_name])[0] # Create parameter ID for grouping
    #     agg_df = trials_df.groupby("param_id")[metric_col].agg(['mean', 'std', 'count']).reset_index()

    #     # Categorical parameter or integer parameter: bar plot
    #     if trials_df[param].dtype == 'object':
    #         colors = get_colors(max_metric, agg_df['mean'])
    #         sns.barplot(x="param_id", y=metric_col, data=trials_df, palette=colors, 
    #             hue="param_id",
    #             legend=False, ax=ax)
    #         ax.set_xlabel("Parameter ID")
    #     else:
    #         #colors = get_colors(max_metric, trials_df[metric_col])
    #         cmap = sns.color_palette(COLORMAP_NAME, as_cmap=True)
    #         sns.scatterplot(x=param, y=metric_col, data=trials_df, palette=cmap,
    #             hue=metric_col, hue_norm=(0, max_metric),
    #             legend=False, ax=ax)
    #         ax.set_xlabel(clean_param_name)

    # #     # Add count labels on bars
    # #     zipper = zip(param_performance['param_id'], param_performance['mean'],param_performance['std'], param_performance['count'])
    # #     for j, (x_pos, mean_val, std_val, count) in enumerate(zipper):
    # #         height = mean_val + std_val
    # #         ax.text(x_pos, height, f'n={count}', ha='center', va='bottom')


    # #     # Put Parameter values in legend
    # #     legend_labels = []
    # #     for j, param_index in enumerate(param_performance.index):
    # #         legend_labels.append(f"{param_index}: {param_performance[param].iloc[j]}")
    # #     ax.legend(legend_labels, title=clean_param_name, bbox_to_anchor=(1.05, 1), loc='upper left')

def main():
    """Main function."""
    args = parse_arguments()
    
    print("="*60)
    print("KERAS TUNER TRIAL ANALYSIS")
    print("="*60)
    print(f"Trial directory: {args.trial_dir}")
    print(f"Project names: {args.project_names if args.project_names else 'All projects'}")
    print(f"Number of best trials: {args.n_best}")
    print(f"Target metric: {args.metric}")
    print(f"Optimization: {'minimize' if args.minimize else 'maximize'}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    project_names: List[str]
    if args.project_names:
        if not isinstance(args.project_names, list):
            project_names = [args.project_names]
        else:
            project_names = args.project_names
    else:
        [os.path.basename(project_name) for project_name in os.listdir(args.trial_dir) if os.path.isdir(os.path.join(args.trial_dir, project_name))]

    all_trials: List[pd.DataFrame] = []
    best_trials: List[pd.DataFrame] = []
    for project_name in project_names:
        print(f"Project name: {project_name}")

        # Load trial data using Keras Tuner API
        trials_data: pd.DataFrame = load_tuner_and_extract_data(
            args.trial_dir, 
            project_name
        )

        # Get best trials
        best_project_trials_df: pd.DataFrame = get_best_trials(
            trials_data, 
            args.metric, 
            args.n_best, 
            args.minimize
        )

        all_trials.append(trials_data)
        best_trials.append(best_project_trials_df)

    print("\nCombining best trials from all projects...")
    print(f"Found total of {sum(len(df) for df in best_trials)} best trials from {len(best_trials)} projects.")

    # Combine trials from all projects
    all_trials_df = pd.concat(all_trials, ignore_index=True)
    best_trials_df = pd.concat(best_trials, ignore_index=True)

    all_trials_df = clean_trials(all_trials_df)
    best_trials_df = clean_trials(best_trials_df)

    best_trials_df = categorize_parameters(best_trials_df, all_trials_df)

    # Determine metric column name
    metric_col = args.metric
    if args.metric not in best_trials_df.columns:
        if f"metric_{args.metric}" in best_trials_df.columns:
            metric_col = f"metric_{args.metric}"
        else:
            metric_col = "score"
    
    # Save to CSV
    csv_path = os.path.join(args.output_dir, f"best_trials.csv")
    save_trials_csv(best_trials_df, csv_path)
    
    # Set style
    sns.set_context("talk")

    if args.grid_search:
        # Create comprehensive plots with all trials
        print("\nCreating comprehensive visualizations (all trials)...")
        plot_trial_ranking(best_trials_df, metric_col, args.output_dir)
        plot_grid_search_analysis(best_trials_df, metric_col, args.output_dir)
        plot_duration_analysis(best_trials_df, metric_col, args.output_dir)
        
        print("\nCreating project-specific visualizations...")
        for project_name in project_names:
            #print(f"\nCreating visualizations for project: {project_name}")
            plot_trial_ranking(best_trials_df, metric_col, args.output_dir, target_project_name=project_name)
            plot_grid_search_analysis(best_trials_df, metric_col, args.output_dir, target_project_name=project_name)
            plot_duration_analysis(best_trials_df, metric_col, args.output_dir, target_project_name=project_name)
    else:
        print("\nCreating Hyperband-specific visualizations...")
        plot_hyperband_analysis(best_trials_df, metric_col, args.output_dir)
        plot_trial_ranking(best_trials_df, metric_col, args.output_dir)
        plot_duration_analysis(best_trials_df, metric_col, args.output_dir, do_sort=True)

        print("\nCreating parameter-specific visualizations...")
        parameter_cols = [col for col in best_trials_df.columns if col.startswith('param_')]
        for project_name in parameter_cols:
            #print(f"\nCreating visualizations for parameter: {project_name}")
            plot_hyperband_analysis(best_trials_df, metric_col, args.output_dir, target_param_name=project_name)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    
    # Print summary statistics
    print(f"\nSummary Statistics for {metric_col}:")
    print(f"Best value: {best_trials_df[metric_col].iloc[0]:.6f}")
    print(f"Worst value: {best_trials_df[metric_col].iloc[-1]:.6f}")
    print(f"Mean: {best_trials_df[metric_col].mean():.6f}")
    print(f"Std: {best_trials_df[metric_col].std():.6f}")
    print(f"Range: {best_trials_df[metric_col].max() - best_trials_df[metric_col].min():.6f}")

if __name__ == "__main__":
    main()