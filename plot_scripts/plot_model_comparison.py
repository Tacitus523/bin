#!/usr/bin/env python3
from typing import Optional
import warnings

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np

# Constants
DPI = 150
FIGURE_SIZE = (8, 6)

PALETTE = sns.color_palette("tab10")
PALETTE.pop(3)  # Remove red color

# Silence seaborn UserWarning about palette length
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"The palette list has more values .* than needed .*",
)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create swarm plots comparing model RMSE values"
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        nargs="+",
        required=True,
        help="Path to input CSV file(s)"
    )
    parser.add_argument(
        "-l", "--labels",
        type=str,
        nargs="+",
        default=None,
        help="Labels for each input file (optional)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="model_comparison_rmse.png",
        help="Output filename (default: model_comparison_rmse.png)"
    )

    args = parser.parse_args()
    return args

def create_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    # Load all input files and add label column
    dfs = []
    for input_file, label in zip(args.input, args.labels):
        df = pd.read_csv(input_file)
        df['dataset'] = label
        dfs.append(df)
        print(f"Loaded {len(df)} rows from {input_file} (label: {label})")
    
    # Concatenate all dataframes
    df = pd.concat(dfs, ignore_index=True)
    
    print(f"\nTotal data: {len(df)} rows")
    print(f"Models: {df['model_name'].unique().tolist()}")
    
    # Prepare data for plotting
    metrics = []
    
    # Energy RMSE
    if 'test_rmse_energy' in df.columns:
        energy_data = df[['model_name', 'model_idx', 'dataset', 'test_rmse_energy']].copy()
        energy_data['metric'] = 'Energy RMSE'
        energy_data['value'] = energy_data['test_rmse_energy']
        energy_data['unit'] = 'eV'
        metrics.append(energy_data[['model_name', 'model_idx', 'dataset', 'metric', 'value', 'unit']])
    
    # Force RMSE
    if 'test_rmse_force' in df.columns:
        force_data = df[['model_name', 'model_idx', 'dataset', 'test_rmse_force']].copy()
        force_data['metric'] = 'Force RMSE'
        force_data['value'] = force_data['test_rmse_force']
        force_data['unit'] = 'eV/Å'
        metrics.append(force_data[['model_name', 'model_idx', 'dataset', 'metric', 'value', 'unit']])
    
    # Charge RMSE
    if 'test_rmse_charge' in df.columns:
        charge_data = df[['model_name', 'model_idx', 'dataset', 'test_rmse_charge']].dropna().copy()
        if len(charge_data) > 0:
            charge_data['metric'] = 'Charge RMSE'
            charge_data['value'] = charge_data['test_rmse_charge']
            charge_data['unit'] = 'e'
            metrics.append(charge_data[['model_name', 'model_idx', 'dataset', 'metric', 'value', 'unit']])
    
    # Combine all metrics
    plot_df = pd.concat(metrics, ignore_index=True)
    return plot_df

def plot_swarm_plots(
        args: argparse.Namespace,
        data: pd.DataFrame
    ) -> None:
    # Count available metrics
    available_metrics = ['Energy RMSE', 'Force RMSE']
    n_metrics = len(available_metrics)
    
    # Create consistent color mapping for all models
    all_models = data['model_name'].unique()
    model_colors = {model: PALETTE[i % len(PALETTE)] for i, model in enumerate(all_models)}
    
    # Create marker mapping for datasets
    all_datasets = data['dataset'].unique()
    n_datasets = len(all_datasets)
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    dataset_markers = {dataset: markers[i % len(markers)] for i, dataset in enumerate(all_datasets)}
    
    # Create subplots
    fig, axes = plt.subplots(1, n_metrics, figsize=(FIGURE_SIZE[0]*n_metrics, FIGURE_SIZE[1]), dpi=DPI)
    if n_metrics == 1:
        axes = [axes]
    
    # Plot each metric
    for idx, metric in enumerate(available_metrics):
        metric_data = data[data['metric'] == metric]
        unit = metric_data['unit'].iloc[0]
        
        ax = axes[idx]
        
        # Plot each dataset separately to apply different markers
        for dataset in metric_data['dataset'].unique():
            dataset_data = metric_data[metric_data['dataset'] == dataset]
            
            # Get colors for models present in this dataset
            models_in_data = dataset_data['model_name'].unique()
            palette = [model_colors[model] for model in models_in_data]

            swarm_plot_kwargs = {
                'data': dataset_data,
                'x': 'model_name',
                'y': 'value',
                'hue': 'model_name',
                'palette': palette,
                'size': 8,
                'ax': ax,
                'legend': False,
                'dodge': False
            }
            if n_datasets > 1:
                swarm_plot_kwargs['marker'] = dataset_markers[dataset]
            
            sns.swarmplot(
                **swarm_plot_kwargs
            )
        
        ax.set_ylim(bottom=0)
        ax.set_xlabel('')
        ax.set_ylabel(f'{metric} ({unit})')
        if n_metrics > 1:
            ax.set_title(metric.replace(" RMSE", ""))
        
        # Rotate x-axis labels if needed
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
        
        # Add horizontal grid for readability
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Add legend for datasets if multiple datasets are present
    if len(all_datasets) > 1:
        legend_elements = [Line2D([0], [0], marker=dataset_markers[dataset], color='gray', 
                                  label=dataset, linestyle='None', markersize=8)
                          for dataset in all_datasets]
        fig.legend(handles=legend_elements, title='Dataset', loc='center left', 
                  bbox_to_anchor=(1.02, 0.5))
    
    plt.savefig(args.output, dpi=DPI, bbox_inches='tight')
    print(f"\nSaved plot to: {args.output}")
    plt.close()

def plot_method_ranking(
    data: pd.DataFrame,
    output_path: str = "charge_method_ranking.png",
    y_label: Optional[str] = None,
    unit: Optional[str] = None
) -> None:
    """
    Create a bar plot showing method rankings based on RMSE.
    
    Args:
        data: DataFrame containing 'model_name' and metric columns
        output_path: Path to save the plot
        unit: Unit for the metric
    """

    # Sort data by the order of the data means
    data = data.copy()
    data_means = data.groupby('model_name')['value'].mean().reset_index().sort_values('value', ascending=True) 
    data['model_name'] = pd.Categorical(
        data['model_name'],
        categories=data_means['model_name'], 
        ordered=True
    )
    data = data.sort_values('model_name').reset_index(drop=True)
    
    # Create color gradient based on metric values
    metric_max = data_means['value'].max()
    norm = plt.Normalize(vmin=0, vmax=metric_max)
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    colors = [cmap(norm(value)) for value in data_means['value']]
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    sns.barplot(
        x='model_name',
        y='value',
        data=data,
        palette=colors,
        hue='model_name',
        legend=False,
        ax=ax
    )
    
    # Set labels
    ax.set_xlabel('Method')
    if unit:
        y_label += f" ({unit})"
    ax.set_ylabel(y_label)
    
    # Rotate x-labels for better readability
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(data_means.iterrows()):
        value = row['value']
        ax.text(i, value, f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved ranking plot to: {output_path}")

def main():
    args = parse_args()
    
    # Validate inputs
    if args.labels is not None and len(args.labels) != len(args.input):
        raise ValueError(f"Number of labels ({len(args.labels)}) must match number of input files ({len(args.input)})")
    
    if args.labels is None:
        args.labels = ["Default"] * len(args.input)
    
    plot_df = create_dataframe(args)

    # Only keep entries with values for the final label, e.g. drop values drop 2G-HDNNP from plot_df if it only has Vacuum label, when Vacuum and Water labels are present
    has_final_label_filter = plot_df['dataset'] == args.labels[-1]
    entries_with_final_label = plot_df[has_final_label_filter]['model_name'].unique()
    plot_df_filter = plot_df['model_name'].isin(entries_with_final_label)
    plot_df = plot_df[plot_df_filter].reset_index(drop=True)
    
    sns.set_context("talk")
    plot_swarm_plots(args, plot_df)
    for metric in plot_df['metric'].unique():
        df_filter = ((plot_df['metric'] == metric) & (plot_df['dataset'] == args.labels[-1])) # Filter to last dataset only
        metric_df = plot_df[df_filter]
        unit = metric_df['unit'].iloc[0]
        plot_method_ranking(
            data=metric_df,
            output_path=f"method_ranking_{metric.replace(' ', '_').lower()}.png",
            y_label=metric,
            unit=unit
        )


if __name__ == "__main__":
    main()
