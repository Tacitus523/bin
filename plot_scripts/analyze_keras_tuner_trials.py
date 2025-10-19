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
COLORMAP_NAME = 'afmhot_r'  
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
        "-n", "--n-best",
        type=int,
        default=15,
        help="Number of best trials to analyze (default: 15)"
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
    args = parser.parse_args()
    args.trial_dir = os.path.abspath(args.trial_dir)
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
        # For the last trial, we can't estimate duration this way, estimate from file modification time
        trial_file = os.path.join(trial_folder, "trial.json")
        if os.path.exists(trial_file):
            mod_time = os.path.getmtime(trial_file)
            duration_seconds = mod_time - current_start.timestamp()
            return duration_seconds if duration_seconds > 0 else None
        else:
            raise FileNotFoundError(f"Trial file not found: {trial_file}")

def load_tuner_and_extract_data(directory: str, project_name: str) -> List[Dict[str, Any]]:
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

    return trial_data

def get_best_trials(trials_data: List[Dict[str, Any]], 
                   metric: str, 
                   n_best: int, 
                   minimize: bool = True) -> pd.DataFrame:
    """Get the best n trials based on the specified metric."""
    
    # Convert to DataFrame
    df = pd.DataFrame(trials_data)
    
    # Check if metric exists
    metric_col = None
    if metric in df.columns:
        metric_col = metric
    elif f"metric_{metric}" in df.columns:
        metric_col = f"metric_{metric}"
    elif "score" in df.columns:
        metric_col = "score"
        print(f"Warning: Metric '{metric}' not found, using 'score' instead")
    else:
        raise ValueError(f"Metric '{metric}' not found in trial data. Available columns: {list(df.columns)}")
    
    # Sort trials by metric
    df_sorted = df.sort_values(metric_col, ascending=minimize)
    
    # Get best n trials
    best_trials = df_sorted.head(n_best).copy()
    
    # Add rank column
    best_trials['rank'] = range(1, len(best_trials) + 1)
    
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
    return trials_df

def save_trials_csv(trials_df: pd.DataFrame, output_path: str) -> None:
    """Save trials data to CSV file."""
    trials_df.to_csv(output_path, index=False)
    print(f"Saved best trials data to: {output_path}")

def create_trial_identifiers(top_rank_df: pd.DataFrame, project_name: Optional[str] = None) -> pd.DataFrame:
    """Create trial identifiers combining project name and trial ID."""
    identifiers = []
    for _, row in top_rank_df.iterrows():
        clean_project_name = row['project_name'].replace('_', ' ').title()
        clean_trial_id = row['trial_id']#f"Trial {row['trial_id']}"
        if project_name is None:
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
    norm = plt.Normalize(vmin=-0.2*metric_max, vmax=metric_max)
    cmap = sns.color_palette(COLORMAP_NAME, as_cmap=True)
    colors = [cmap(norm(value)) for value in values]
    return colors


def plot_hyperparameter_analysis(trials_df: pd.DataFrame, 
                                metric_col: str,
                                output_dir: str,
                                project_name: Optional[str] = None ) -> None:
    """Create bar plots for hyperparameter analysis."""

    # Normalize metric values for colormap
    metric_max = trials_df[metric_col].max()

    # Create colormap between lowest and highest metric values
    if project_name is not None:
        trials_df = trials_df[trials_df['project_name'] == project_name]

    
    trials_df = trials_df.dropna(how='all', axis=1)
    trials_df = trials_df.sort_values(["project_name", "trial_id"])

    #hyperparam_cols = [col for col in trials_df.columns if col.startswith('param_')]
    project_names = trials_df["project_name"].unique().tolist()
    
    # Create subplots
    n_projects = len(project_names)
    n_cols = min(3, n_projects)
    n_rows = (n_projects + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_projects == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if n_projects > 1 else [axes]
    else:
        axes = axes.flatten()
    
    for i, param in enumerate(project_names):
        project_df = trials_df[trials_df["project_name"] == param]
        clean_project_name = param.replace('param_', '').replace('_', ' ').title()
        #clean_metric_name = metric_col.replace('metric_', '').replace('_', ' ').title()
        ax = axes[i] if i < len(axes) else None
        if ax is None:
            continue
            

        # # Categorical parameter - bar plot
        # param_performance = trials_df.groupby(param)[metric_col].agg(['mean', 'std', 'count'])
        
        if False: #np.any(param_performance['count'] > 1):
            bars = ax.bar(range(len(param_performance)), param_performance['mean'], 
                            yerr=param_performance['std'], capsize=5, alpha=0.7)
            ax.set_xlabel(param)
            ax.set_ylabel(f'Mean {SCORE_LABEL}')
            ax.set_title(f'{param} Performance')
            ax.set_xticks(range(len(param_performance)))
            ax.set_xticklabels(param_performance.index, rotation=45)
            
            # Add count labels on bars
            for j, (bar, count) in enumerate(zip(bars, param_performance['count'])):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'n={count}', ha='center', va='bottom')
        else:
            #param_df = trials_df.dropna(subset=[param, metric_col])
            # Create colors based on metric values
            colors = get_colors(metric_max, project_df[metric_col])
            sns.barplot(x="trial_id", y=metric_col, data=project_df, ax=ax, palette=colors, hue="trial_id", legend=False)
            ax.set_title(f'Performance {clean_project_name}' if project_name is None else None)
            ax.set_xlabel('Trial ID')
            ax.set_ylabel(SCORE_LABEL)
            ax.set_ylim(0, metric_max*1.1)
    
    # Hide unused subplots
    for i in range(n_projects, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    filename = f"hyperparameter_analysis_{project_name}.png" if project_name else "hyperparameter_analysis.png"
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

    top_rank_df = trials_df.sort_values(metric_col).head(10).reset_index(drop=True)
    
    plt.figure(figsize=(12, 6))
    
    # Create bar plot with colors based on metric values
    colors = get_colors(max_metric, top_rank_df[metric_col])

    # Use the function
    top_rank_df = create_trial_identifiers(top_rank_df, target_project_name)
    
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
                          target_project_name: Optional[str] = None ) -> None:
    """Create plots analyzing trial duration vs performance."""

    max_metric = trials_df[metric_col].max()

    trials_df = trials_df.sort_values(["project_name", "trial_id"])

    if target_project_name is not None:
        trials_df = trials_df[trials_df['project_name'] == target_project_name]
    
    possible_timing_keys = ['total_training_time', 'duration', 'elapsed_time', 'elapsed_process_time', 'total_process_time']
    if not any(key in trials_df.columns for key in possible_timing_keys):
        print("Warning: No timing information found in trials data")
        return

    possible_real_time_keys = ['total_training_time', 'duration', 'elapsed_time']
    for key in possible_real_time_keys:
        if key in trials_df.columns:
            real_time_key = key
            break
    possible_process_time_keys = ['elapsed_process_time', 'total_process_time'] # total_process_time seems bugged, way too high
    for key in possible_process_time_keys:
        if key in trials_df.columns:
            process_time_key = key
            break

    # Filter out NaN durations
    valid_duration_df = trials_df.dropna(subset=[real_time_key, process_time_key])
    
    if len(valid_duration_df) == 0:
        print("Warning: No valid duration data found")
        return
    
    project_names = trials_df["project_name"].unique().tolist()
    
    # Create subplots
    n_projects = len(project_names)
    n_cols = min(3, n_projects)
    n_rows = (n_projects + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_projects == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if n_projects > 1 else [axes]
    else:
        axes = axes.flatten()
    
    for i, project_name in enumerate(project_names):
        project_df = valid_duration_df[valid_duration_df["project_name"] == project_name]
        clean_project_name = project_name.replace('_', ' ').title()
        ax = axes[i] if i < len(axes) else None
        if ax is None:
            continue

        project_df = create_trial_identifiers(project_df, project_name)

        colors = get_colors(max_metric, project_df[metric_col])
        sns.barplot(x='identifier', y=real_time_key, data=project_df, 
            palette=colors, hue='identifier', legend=False, ax=ax)
        ax.set_title(clean_project_name if target_project_name is None else None)
        ax.set_xlabel('Trial ID')
        ax.set_ylabel('Duration (minutes)')

    # Hide unused subplots
    for i in range(n_projects, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    filename = f"duration_analysis_{target_project_name}.png" if target_project_name else "duration_analysis.png"
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path, dpi=DPI, bbox_inches='tight')
    plt.close()

    if target_project_name is None:
        create_duration_overview_plot(valid_duration_df, metric_col, real_time_key, process_time_key, output_dir)

def create_duration_overview_plot(duration_df: pd.DataFrame, 
                                    metric_col: str,
                                    real_time_key: str,
                                    process_time_key: str,
                                    output_dir: str) -> None:

        max_metric = duration_df[metric_col].max()

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Duration vs Performance scatter plot
        colors = get_colors(max_metric, duration_df[metric_col])
        sns.scatterplot(x=real_time_key, y=metric_col, hue=real_time_key, 
            data=duration_df, palette=colors, legend=False,
            alpha=0.7, s=100, ax=axes[0])
        axes[0].set_xlabel('Duration (minutes)')
        axes[0].set_ylabel(SCORE_LABEL)
        axes[0].set_title('Trial Duration vs Performance')
        
        # Duration distribution
        sns.histplot(data=duration_df, x=real_time_key, bins=min(20, len(duration_df)),
                    alpha=0.7, edgecolor='black', ax=axes[1])
        axes[1].set_xlabel('Duration (minutes)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Trial Duration Distribution')
        axes[1].axvline(duration_df[real_time_key].mean(), 
                    color='red', linestyle='--', 
                    label=f'Mean: {duration_df[real_time_key].mean():.1f} min')
        axes[1].legend()
        
        plt.tight_layout()
        
        # Save plot
        filename = "duration_overview.png"
        plot_path = os.path.join(output_dir, filename)
        plt.savefig(plot_path, dpi=DPI, bbox_inches='tight')
        plt.close()
    
        # Print duration statistics
        print(f"\nDuration Statistics:")
        print(f"Mean duration: {duration_df[real_time_key].mean():.1f} minutes")
        print(f"Median duration: {duration_df[real_time_key].median():.1f} minutes")
        print(f"Min duration: {duration_df[real_time_key].min():.1f} minutes")
        print(f"Max duration: {duration_df[real_time_key].max():.1f} minutes")

        print(f"Process Time Statistics:")
        print(f"Mean process time: {duration_df[process_time_key].mean():.1f} minutes")
        print(f"Median process time: {duration_df[process_time_key].median():.1f} minutes")
        print(f"Min process time: {duration_df[process_time_key].min():.1f} minutes")
        print(f"Max process time: {duration_df[process_time_key].max():.1f} minutes")

def main():
    """Main function."""
    args = parse_arguments()
    
    print("="*60)
    print("KERAS TUNER TRIAL ANALYSIS")
    print("="*60)
    print(f"Number of best trials: {args.n_best}")
    print(f"Target metric: {args.metric}")
    print(f"Optimization: {'minimize' if args.minimize else 'maximize'}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    best_trials: List[pd.DataFrame] = []
    for project_name in os.listdir(args.trial_dir):
        print(f"Project name: {project_name}")

        # Load trial data using Keras Tuner API
        trials_data: List[Dict[str, Any]] = load_tuner_and_extract_data(
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
        
        best_trials.append(best_project_trials_df)

    print("\nCombining best trials from all projects...")
    print(f"Found total of {sum(len(df) for df in best_trials)} best trials from {len(best_trials)} projects.")

    # Combine best trials from all projects
    best_trials_df = pd.concat(best_trials, ignore_index=True)

    best_trials_df = clean_trials(best_trials_df)

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

    # Create comprehensive plots with all trials
    print("\nCreating comprehensive visualizations (all trials)...")
    plot_trial_ranking(best_trials_df, metric_col, args.output_dir)
    plot_hyperparameter_analysis(best_trials_df, metric_col, args.output_dir)
    plot_duration_analysis(best_trials_df, metric_col, args.output_dir)
    
    print("\nCreating project-specific visualizations...")
    for project_name in best_trials_df['project_name'].unique():
        print(f"\nCreating visualizations for project: {project_name}")
        plot_trial_ranking(best_trials_df, metric_col, args.output_dir, target_project_name=project_name)
        plot_hyperparameter_analysis(best_trials_df, metric_col, args.output_dir, project_name=project_name)
        plot_duration_analysis(best_trials_df, metric_col, args.output_dir, target_project_name=project_name)
    
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