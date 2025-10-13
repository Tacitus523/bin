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

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze Keras Tuner trials and create visualizations"
    )
    parser.add_argument(
        "trial_dir",
        type=str,
        help="Path to Keras Tuner trial directory"
    )
    parser.add_argument(
        "-n", "--n-best",
        type=int,
        default=10,
        help="Number of best trials to analyze (default: 10)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=".",
        help="Output directory for CSV and plots (default: current directory)"
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
    args.directory = os.path.dirname(args.trial_dir)
    args.project_name = os.path.basename(args.trial_dir)
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
    
    print(f"Loading Keras Tuner from directory: {directory}")
    print(f"Project name: {project_name}")
    try:
        tuner = DummyTuner(
            directory=directory,
            project_name=project_name,
            overwrite=False  # Don't overwrite existing trials
        )
        
        if len(tuner.oracle.trials) == 0:
            raise ValueError("No trials found in the specified directory and project name.")
        print(f"Found {len(tuner.oracle.trials)} trials")
        
    except Exception as e:
        print(f"Error creating tuner: {e}")
        raise
    
    # Extract timing information from oracle.json
    trial_start_times: List[datetime] = tuner.oracle._display.trial_start
    
    # Extract trial data using tuner API
    trials_data = []
    
    for trial_id, trial in tuner.oracle.trials.items():
        # Extract basic trial information
        trial_data = {
            'trial_id': trial_id,
            'status': trial.status,
            'score': trial.score
        }
        
        # Extract hyperparameters
        for hp_name, hp_value in trial.hyperparameters.values.items():
            if hp_name.startswith('tuner/'):
                continue
            trial_data[hp_name] = hp_value
        
        # Extract metrics
        if trial.metrics:
            for metric_name, metric_observations in trial.metrics.metrics.items():
                best_obs = metric_observations.get_best_value()
                if best_obs is not None:
                    trial_data[f"metric_{metric_name}"] = best_obs
        
        # Extract timing information
        if trial_id in trial_start_times:
            trial_folder = tuner.get_trial_dir(trial_id)
            duration = calculate_trial_duration(trial_id, trial_start_times, trial_folder)
            trial_data['duration'] = duration // 60 # Convert to minutes
        
        trials_data.append(trial_data)
            
    
    # Print summary statistics
    trials_with_duration = sum(1 for trial in trials_data if trial.get('duration') is not None)
    print(f"Processed {len(trials_data)} trials")
    print(f"Duration data available for {trials_with_duration}/{len(trials_data)} trials")
    print("Warning: No garantee that duration estimates are accurate. Early stopping and hyperband scheduling can affect this.")
    
    if trials_with_duration > 0:
        durations = [trial['duration'] for trial in trials_data if trial.get('duration') is not None]
        avg_duration = sum(durations) / len(durations)
        print(f"Average trial duration: {avg_duration:.1f} minutes")
    
    return trials_data

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
    
    print(f"Best {len(best_trials)} trials based on {metric_col}:")
    print(f"Best score: {best_trials[metric_col].iloc[0]:.6f}")
    print(f"Worst score in top {n_best}: {best_trials[metric_col].iloc[-1]:.6f}")
    
    return best_trials

def save_trials_csv(trials_df: pd.DataFrame, output_path: str) -> None:
    """Save trials data to CSV file."""
    trials_df.to_csv(output_path, index=False)
    print(f"Saved best trials data to: {output_path}")

def plot_hyperparameter_analysis(trials_df: pd.DataFrame, 
                                metric_col: str,
                                output_dir: str) -> None:
    """Create bar plots for hyperparameter analysis."""
    
    # Set style
    sns.set_context("talk")
    plt.style.use('default')
    
    # Get hyperparameter columns (exclude metadata columns)
    exclude_cols = {'trial_id', 'score', 'rank', 'duration', 'status'} | {col for col in trials_df.columns if col.startswith('metric_')}
    hyperparam_cols = [col for col in trials_df.columns if col not in exclude_cols]
    
    if not hyperparam_cols:
        print("Warning: No hyperparameter columns found for plotting")
        return
    
    # Create subplots
    n_params = len(hyperparam_cols)
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_params == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if n_params > 1 else [axes]
    else:
        axes = axes.flatten()
    
    for i, param in enumerate(hyperparam_cols):
        ax = axes[i] if i < len(axes) else None
        if ax is None:
            continue
            
        # Check if parameter is numeric or categorical
        if pd.api.types.is_numeric_dtype(trials_df[param]):
            # Numeric parameter - scatter plot
            scatter = ax.scatter(trials_df[param], trials_df[metric_col], 
                               c=trials_df['rank'], cmap='viridis_r', 
                               alpha=0.7, s=100)
            ax.set_xlabel(param)
            ax.set_ylabel(metric_col)
            ax.set_title(f'{param} vs {metric_col}')
            plt.colorbar(scatter, ax=ax, label='Rank')
        else:
            # Categorical parameter - bar plot
            param_performance = trials_df.groupby(param)[metric_col].agg(['mean', 'std', 'count'])
            
            bars = ax.bar(range(len(param_performance)), param_performance['mean'], 
                         yerr=param_performance['std'], capsize=5, alpha=0.7)
            ax.set_xlabel(param)
            ax.set_ylabel(f'Mean {metric_col}')
            ax.set_title(f'{param} Performance')
            ax.set_xticks(range(len(param_performance)))
            ax.set_xticklabels(param_performance.index, rotation=45)
            
            # Add count labels on bars
            for j, (bar, count) in enumerate(zip(bars, param_performance['count'])):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'n={count}', ha='center', va='bottom')
    
    # Hide unused subplots
    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, "hyperparameter_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved hyperparameter analysis plot to: {plot_path}")

def plot_trial_ranking(trials_df: pd.DataFrame, 
                      metric_col: str,
                      output_dir: str) -> None:
    """Create a bar plot showing trial rankings."""
    
    plt.figure(figsize=(12, 6))
    
    # Create bar plot
    bars = plt.bar(range(len(trials_df)), trials_df[metric_col], 
                   color=sns.color_palette("viridis", len(trials_df)))
    
    plt.xlabel('Trial Rank')
    plt.ylabel(metric_col)
    plt.title(f'Best {len(trials_df)} Trials - {metric_col}')
    
    # Add trial IDs as labels
    plt.xticks(range(len(trials_df)), 
               [f"{i+1}" for i in range(len(trials_df))], 
               rotation=45)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, trials_df[metric_col])):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, "trial_ranking.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved trial ranking plot to: {plot_path}")

def plot_duration_analysis(trials_df: pd.DataFrame, 
                          metric_col: str,
                          output_dir: str) -> None:
    """Create plots analyzing trial duration vs performance."""
    
    if 'duration' not in trials_df.columns or trials_df['duration'].isna().all():
        print("Warning: No duration data available for analysis")
        return
    
    # Filter out NaN durations
    valid_duration_df = trials_df.dropna(subset=['duration']).copy()
    
    if len(valid_duration_df) == 0:
        print("Warning: No valid duration data found")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Duration vs Performance scatter plot
    scatter = axes[0].scatter(valid_duration_df['duration'], 
                             valid_duration_df[metric_col],
                             c=valid_duration_df['rank'], 
                             cmap='viridis_r', alpha=0.7, s=100)
    axes[0].set_xlabel('Duration (minutes)')
    axes[0].set_ylabel(metric_col)
    axes[0].set_title('Trial Duration vs Performance')
    plt.colorbar(scatter, ax=axes[0], label='Rank')
    
    # Duration distribution
    axes[1].hist(valid_duration_df['duration'], bins=min(20, len(valid_duration_df)), 
                alpha=0.7, color='skyblue', edgecolor='black')
    axes[1].set_xlabel('Duration (minutes)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Trial Duration Distribution')
    axes[1].axvline(valid_duration_df['duration'].mean(), 
                   color='red', linestyle='--', 
                   label=f'Mean: {valid_duration_df["duration"].mean():.1f} min')
    axes[1].legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, "duration_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved duration analysis plot to: {plot_path}")
    
    # Print duration statistics
    print(f"\nDuration Statistics:")
    print(f"Mean duration: {valid_duration_df['duration'].mean():.1f} minutes")
    print(f"Median duration: {valid_duration_df['duration'].median():.1f} minutes")
    print(f"Min duration: {valid_duration_df['duration'].min():.1f} minutes")
    print(f"Max duration: {valid_duration_df['duration'].max():.1f} minutes")

def main():
    """Main function."""
    args = parse_arguments()
    
    print("="*60)
    print("KERAS TUNER TRIAL ANALYSIS")
    print("="*60)
    print(f"Directory: {args.directory}")
    print(f"Project name: {args.project_name}")
    print(f"Number of best trials: {args.n_best}")
    print(f"Target metric: {args.metric}")
    print(f"Optimization: {'minimize' if args.minimize else 'maximize'}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load trial data using Keras Tuner API
    trials_data = load_tuner_and_extract_data(
        args.directory, 
        args.project_name
    )
    
    # Get best trials
    best_trials_df = get_best_trials(
        trials_data, 
        args.metric, 
        args.n_best, 
        args.minimize
    )
    
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
    
    # Create plots
    print("\nCreating visualizations...")
    plot_trial_ranking(best_trials_df, metric_col, args.output_dir)
    plot_hyperparameter_analysis(best_trials_df, metric_col, args.output_dir)
    plot_duration_analysis(best_trials_df, metric_col, args.output_dir)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    
    # Print summary statistics
    print(f"\nSummary Statistics for {metric_col}:")
    print(f"Best value: {best_trials_df[metric_col].iloc[0]:.6f}")
    print(f"Mean: {best_trials_df[metric_col].mean():.6f}")
    print(f"Std: {best_trials_df[metric_col].std():.6f}")
    print(f"Range: {best_trials_df[metric_col].max() - best_trials_df[metric_col].min():.6f}")

if __name__ == "__main__":
    main()