#!/usr/bin/env python3

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Constants
DPI = 150
FIGURE_SIZE = (12, 5)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create swarm plots comparing model RMSE values"
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Path to input CSV file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="model_comparison_rmse.png",
        help="Output filename (default: model_comparison_rmse.png)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Read the CSV file
    df = pd.read_csv(args.input)
    
    print(f"Loaded data with {len(df)} rows")
    print(f"Models: {df['model_name'].unique().tolist()}")
    
    # Prepare data for plotting
    metrics = []
    
    # Energy RMSE
    if 'test_rmse_energy' in df.columns:
        energy_data = df[['model_name', 'model_idx', 'test_rmse_energy']].copy()
        energy_data['metric'] = 'Energy RMSE'
        energy_data['value'] = energy_data['test_rmse_energy']
        energy_data['unit'] = 'eV'
        metrics.append(energy_data[['model_name', 'model_idx', 'metric', 'value', 'unit']])
    
    # Force RMSE
    if 'test_rmse_force' in df.columns:
        force_data = df[['model_name', 'model_idx', 'test_rmse_force']].copy()
        force_data['metric'] = 'Force RMSE'
        force_data['value'] = force_data['test_rmse_force']
        force_data['unit'] = 'eV/Å'
        metrics.append(force_data[['model_name', 'model_idx', 'metric', 'value', 'unit']])
    
    # Charge RMSE
    if 'test_rmse_charge' in df.columns:
        charge_data = df[['model_name', 'model_idx', 'test_rmse_charge']].dropna().copy()
        if len(charge_data) > 0:
            charge_data['metric'] = 'Charge RMSE'
            charge_data['value'] = charge_data['test_rmse_charge']
            charge_data['unit'] = 'e'
            metrics.append(charge_data[['model_name', 'model_idx', 'metric', 'value', 'unit']])
    
    # Combine all metrics
    plot_df = pd.concat(metrics, ignore_index=True)
    
    # Count available metrics
    available_metrics = plot_df['metric'].unique()
    n_metrics = len(available_metrics)
    
    print(f"\nAvailable metrics: {available_metrics.tolist()}")
    
    # Create consistent color mapping for all models
    all_models = df['model_name'].unique()
    colors = sns.color_palette('tab10', n_colors=len(all_models))
    model_colors = {model: colors[i] for i, model in enumerate(all_models)}
    
    print(f"Color mapping: {list(model_colors.keys())}")
    
    # Create subplots
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
    
    sns.set_context("talk")
    
    # Plot each metric
    for idx, metric in enumerate(available_metrics):
        metric_data = plot_df[plot_df['metric'] == metric]
        unit = metric_data['unit'].iloc[0]
        
        # Get colors for models present in this metric
        models_in_metric = metric_data['model_name'].unique()
        palette = [model_colors[model] for model in models_in_metric]
        
        ax = axes[idx]
        sns.swarmplot(
            data=metric_data,
            x='model_name',
            y='value',
            hue='model_name',
            palette=palette,
            size=8,
            ax=ax,
            legend=False
        )
        
        ax.set_ylim(bottom=0)
        ax.set_xlabel('')
        ax.set_ylabel(f'{metric} ({unit})')
        ax.set_title(metric.replace(" RMSE", ""))
        
        # Rotate x-axis labels if needed
        ax.tick_params(axis='x', rotation=45)
        
        # Add horizontal grid for readability
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Print statistics
        print(f"\n{metric}:")
        for model in metric_data['model_name'].unique():
            model_values = metric_data[metric_data['model_name'] == model]['value']
            print(f"  {model}: mean={model_values.mean():.6f}, std={model_values.std():.6f}")
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=DPI, bbox_inches='tight')
    print(f"\nSaved plot to: {args.output}")
    plt.close()

if __name__ == "__main__":
    main()
