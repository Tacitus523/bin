#!/usr/bin/env python
"""
Script to plot ESP reproducibility by charge type from extxyz files.
Extracts ESP_ properties from arrays and E_elec_ properties from info,
and creates scatter plots comparing charge density to other methods.
"""

import argparse
from typing import Dict, List, Optional, Union, Tuple
from ase.atoms import Atoms
from ase.io import read
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.metrics import mean_squared_error, r2_score

N_SUBSAMPLE = 10000
ESP_CUTOFF = 2.5 # Some ESP values got suspiciously positive
FIGSIZE = (8, 6)
ALPHA = 0.3
SIZE = 20
DPI = 100

ESP_UNIT = "eV/e"
ENERGY_UNIT = "eV"

KEY_LABEL_MAP = {
    "ESP_charge_density": r"$\text{ESP}_\text{Charge density}$",
    "ESP_charges_hirsh": r"$\text{ESP}_\text{Hirshfeld}$",
    "ESP_charges_mull": r"$\text{ESP}_\text{Mulliken}$",
    "ESP_charges_loew": r"$\text{ESP}_\text{Löwdin}$",
    "ESP_charges_chelpg": r"$\text{ESP}_\text{CHELPG}$",
    "ESP_charges_mk": r"$\text{ESP}_\text{MK}$",
    "ESP_charges_resp": r"$\text{ESP}_\text{RESP}$",
    "ESP_charges_eem": r"$\text{ESP}_\text{EEM}$",
    "ESP_charges_mbis": r"$\text{ESP}_\text{MBIS}$",
    "E_elec_charge_density": r"$E_\text{elec, Charge density}$",
    "E_elec_charges_hirsh": r"$E_\text{elec, Hirshfeld}$",
    "E_elec_charges_mull": r"$E_\text{elec, Mulliken}$",
    "E_elec_charges_loew": r"$E_\text{elec, Löwdin}$",
    "E_elec_charges_chelpg": r"$E_\text{elec, CHELPG}$",
    "E_elec_charges_mk": r"$E_\text{elec, MK}$",
    "E_elec_charges_resp": r"$E_\text{elec, RESP}$",
    "E_elec_charges_eem": r"$E_\text{elec, EEM}$",
    "E_elec_charges_mbis": r"$E_\text{elec, MBIS}$",
    "hirsh": "Hirshfeld",
    "mull": "Mulliken",
    "loew": "Löwdin",
    "chelpg": "CHELPG",
    "mk": "MK",
    "resp": "RESP",
    "eem": "EEM",
    "mbis": "MBIS",
}

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Plot ESP reproducibility by charge type")
    parser.add_argument(
        "-f", "--file",
        type=str,
        required=True,
        help="Path to the extxyz file containing ESP and E_elec data"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="esp_plots",
        help="Output directory for plots (default: esp_plots)"
    )
    args = parser.parse_args()
    return args

def extract_esp_properties(atoms_list: List[Atoms]) -> pd.DataFrame:
    """
    Extract ESP_ properties from atoms.arrays and E_elec_ properties from atoms.info.
    
    Args:
        atoms_list: List of ASE Atoms objects
        
    Returns:
        pd.DataFrame containing extracted properties
    """
    data = {}
    sources = []
    
    # Get all unique ESP_ and E_elec_ keys from the first atom
    esp_keys = [key for key in atoms_list[0].arrays.keys() if key.startswith('ESP_')]
    elec_keys = [key for key in atoms_list[0].info.keys() if key.startswith('E_elec_')]
    
    print(f"Found ESP properties in arrays: {esp_keys}")
    print(f"Found E_elec properties in info: {elec_keys}")
    
    # Initialize data dictionary
    for key in esp_keys:
        data[key] = []
    for key in elec_keys:
        data[key] = []
    
    data['molecule_index'] = []
    data['atom_index'] = []

    # Extract data from all atoms
    for mol_idx, atoms in enumerate(atoms_list):
        n_atoms = len(atoms)
        # Extract ESP properties from arrays (per-atom properties)
        for key in esp_keys:
            data[key].extend(atoms.arrays[key].flatten())
        
        # Extract E_elec properties from info (per-structure properties)
        for key in elec_keys:
            # Repeat the value for each atom in the structure
            data[key].extend([atoms.info[key]] * n_atoms)
        
        # Track molecule and atom indices
        data['molecule_index'].extend([mol_idx] * n_atoms)
        data['atom_index'].extend(list(range(n_atoms)))

        # Extract source information if available
        if 'source' in atoms.info:
            sources.extend([atoms.info['source']] * n_atoms)
        else:
            sources.extend(['unknown'] * n_atoms)
    
    # Convert lists to numpy arrays
    for key in data:
        data[key] = np.array(data[key])
    
    # Add sources if available
    if sources and any(s != 'unknown' for s in sources):
        data['source'] = np.array(sources)
    
    data = pd.DataFrame(data)

    return data

def calculate_metrics(x: np.ndarray, y: np.ndarray) -> Tuple:
    """Calculate RMSE and R² metrics."""
    rmse = mean_squared_error(x, y, squared=False)
    r2 = r2_score(x, y)
    return rmse, r2

def create_scatter_plot(
    data: pd.DataFrame,
    x_key: str,
    y_key: str,
    output_path: str,
    min_max: Optional[Tuple[float, float]] = None,
    unit: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[float, float]:
    """
    Create a scatter plot comparing two properties.
    
    Args:
        data: DataFrame containing the data
        x_key: Key for x-axis data
        y_key: Key for y-axis data
        output_path: Path to save the plot
        min_max: Optional tuple of (min, max) for axis limits
        unit: Unit for the data
        title: Optional title for the plot
        ax: Optional axes to plot on (if None, creates new figure and saves it)
        
    Returns:
        Tuple of (RMSE, R²) metrics
    """
    if x_key not in data or y_key not in data:
        raise ValueError(f"Keys {x_key} and/or {y_key} not found in data.")
    
    x_data = data[x_key]
    y_data = data[y_key]
    
    # Calculate metrics
    rmse, r2 = calculate_metrics(x_data, y_data)
    
    # Add source information if available
    if 'source' in data:
        hue = 'source'
    else:
        hue = None
    
    # Subsample data for plotting if too large
    random_indices = np.random.choice(len(x_data), size=min(len(x_data), N_SUBSAMPLE), replace=False)
    random_indices.sort()
    subsampled_data = data.iloc[random_indices]

    # Create figure if needed
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=FIGSIZE)
    
    # Create scatter plot
    sns.scatterplot(
        data=subsampled_data,
        x=x_key,
        y=y_key,
        hue=hue,
        palette="tab10" if hue else None,
        alpha=ALPHA,
        edgecolor=None,
        s=SIZE,
        ax=ax
    )
    
    # Add identity line
    if min_max:
        min_val, max_val = min_max
    else:
        min_val = min(np.min(x_data), np.min(y_data))
        max_val = max(np.max(x_data), np.max(y_data))
    ax.plot([min_val, max_val], [min_val, max_val], color="black", linestyle="--", alpha=0.8, linewidth=1)
    
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    # Set labels and title
    if x_key not in KEY_LABEL_MAP:
        print(f"Warning: No label found for key '{x_key}' in KEY_LABEL_MAP.")
    if y_key not in KEY_LABEL_MAP:
        print(f"Warning: No label found for key '{y_key}' in KEY_LABEL_MAP.")
    x_label = KEY_LABEL_MAP.get(x_key, x_key)
    y_label = KEY_LABEL_MAP.get(y_key, y_key)
    if not standalone:
        ax.set_title(f"{y_label}")
    if unit:
        x_label += f" ({unit})"
        y_label += f" ({unit})"
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    # Add metrics text box
    ax.text(
        0.65,
        0.25,
        f"RMSE: {rmse:.3f} {unit or ' '}\nR²: {r2:.4f}",
        transform=ax.transAxes,
        fontsize=15,
        verticalalignment="top",
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
    )
    
    # Improve legend if available
    if hue:
        ax.legend(title=None, loc="upper left", fontsize="x-small")
        for legend_handle in ax.get_legend().legend_handles:
            legend_handle.set_alpha(1.0)
    
    # Save and close if standalone
    if standalone:
        plt.tight_layout()
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"{x_key} vs {y_key}:")
        print(f"  RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
    return rmse, r2

def create_scatter_plots(
        data: pd.DataFrame,
        x_key: str,
        y_keys: List[str],
        output_dir: str,
        unit: Optional[str] = None
    ) -> pd.DataFrame:

    n_charge_types = len(y_keys)
    n_cols = 2
    n_rows = (n_charge_types + n_cols - 1) // n_cols  # Round up to the nearest whole number

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * FIGSIZE[0], n_rows * FIGSIZE[1]))
    axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing

    # get 1st and 99th percentiles for axis limits
    # to avoid outliers affecting the axis limits too much
    low_quantile = data[[x_key]+y_keys].quantile(0.01).min()
    high_quantile = data[[x_key]+y_keys].quantile(0.99).max()

    metrics_df: pd.DataFrame = pd.DataFrame(columns=['y_key', 'RMSE', 'R2'])
    for i, y_key in enumerate(y_keys):
        output_file = os.path.join(output_dir, f"{x_key}_vs_{y_key.replace('ESP_', '').replace('E_elec_', '')}.png")

        rmse,r2 = create_scatter_plot(
            data,
            x_key,
            y_key,
            output_file,
            min_max=(low_quantile, high_quantile),
            unit=unit,
        )
        create_scatter_plot(
            data,
            x_key,
            y_key,
            "",
            min_max=(low_quantile, high_quantile),
            unit=unit,
            ax=axes[i]
        )
        metrics_df.loc[i] = {'y_key': y_key, 'RMSE': rmse, 'R2': r2}

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    combined_output_file = os.path.join(output_dir, f"{x_key}_vs_all.png")
    plt.savefig(combined_output_file, dpi=DPI, bbox_inches='tight')
    plt.close()

    return metrics_df

def filter_data_by_cutoff(
    data: pd.DataFrame,
    key: str,
    cutoff: float
) -> pd.DataFrame:
    """Filter data to exclude entries where the absolute value of key exceeds cutoff."""
    filtered_data = data[data[key] <= cutoff].copy()
    n_excluded = len(data) - len(filtered_data)
    if n_excluded > 0:
        print(f"Excluded {n_excluded} entries from '{key}' exceeding cutoff of {cutoff}.")
    return filtered_data

def plot_method_ranking(
    metrics_df: pd.DataFrame,
    output_path: str,
    metric_name: str = "RMSE",
    y_label: Optional[str] = None,
    unit: Optional[str] = None
) -> None:
    """
    Create a bar plot showing method rankings based on RMSE.
    
    Args:
        metrics_df: DataFrame containing y_key, RMSE, and R2 columns
        output_path: Path to save the plot
        metric_name: Name of the metric to plot (default: "RMSE")
        unit: Unit for the metric
    """
    if metrics_df.empty:
        print("Warning: Empty metrics dataframe, skipping ranking plot")
        return
    
    # Sort by metric (ascending for RMSE, descending for R2)
    ascending = metric_name == "RMSE"
    sorted_df = metrics_df.sort_values(metric_name, ascending=ascending).reset_index(drop=True)
    
    # Create clean labels for methods
    clean_labels = []
    for y_key in sorted_df['y_key']:
        # Remove the prefix and extract just the method name
        if 'ESP' in y_key:
            method = y_key.replace('ESP_charges_', '').replace('ESP_', '')
            method = KEY_LABEL_MAP[method]
        elif 'E_elec' in y_key:
            method = y_key.replace('E_elec_charges_', '').replace('E_elec_', '')
            method = KEY_LABEL_MAP[method]
        else:
            raise ValueError(f"Unexpected y_key format: {y_key}")
        clean_labels.append(method)

    sorted_df['method'] = clean_labels
    
    # Create color gradient based on metric values
    metric_max = sorted_df[metric_name].max()
    norm = plt.Normalize(vmin=0, vmax=metric_max)
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    colors = [cmap(norm(value)) for value in sorted_df[metric_name]]
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.barplot(
        x='method',
        y=metric_name,
        data=sorted_df,
        palette=colors,
        hue='method',
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
    for i, (idx, row) in enumerate(sorted_df.iterrows()):
        value = row[metric_name]
        ax.text(i, value, f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved ranking plot to: {output_path}")

def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load the extxyz file
    print(f"Loading {args.file}...")
    atoms_list = read(args.file, format="extxyz", index=":")
    print(f"Loaded {len(atoms_list)} structures")
    
    # Extract properties
    data: pd.DataFrame = extract_esp_properties(atoms_list)
    data = filter_data_by_cutoff(data, 'ESP_charge_density', ESP_CUTOFF)

    # Print data summary
    print("\nData summary:")
    print(f"  Total entries: {len(data)}")
    
    # Find charge_density reference
    esp_charge_density = "ESP_charge_density"
    elec_charge_density = "E_elec_charge_density"
    
    print(f"\nReference keys:")
    print(f"  ESP charge density: {esp_charge_density}")
    print(f"  E_elec charge density: {elec_charge_density}")

    sns.set_context("talk")
    
    # Create plots comparing ESP_charge_density to other ESP methods
    esp_keys = [key for key in data.keys() if key.startswith('ESP_') and key != esp_charge_density]
    esp_metrics = create_scatter_plots(
        data, 
        esp_charge_density, 
        esp_keys, 
        args.output,
        unit=ESP_UNIT
    )
    
    # Create plots comparing E_elec_charge_density to other E_elec methods
    e_elec_data = data.groupby('molecule_index').first().reset_index()
    elec_keys = [key for key in data.keys() if key.startswith('E_elec_') and key != elec_charge_density]
    e_elec_metrics = create_scatter_plots(
        e_elec_data, 
        elec_charge_density, 
        elec_keys, 
        args.output,
        unit=ENERGY_UNIT
    )
    
    # Save metrics to CSV
    esp_csv_path = os.path.join(args.output, "esp_metrics.csv")
    esp_metrics.to_csv(esp_csv_path, index=False)
    print(f"\nSaved ESP metrics to: {esp_csv_path}")
    
    e_elec_csv_path = os.path.join(args.output, "e_elec_metrics.csv")
    e_elec_metrics.to_csv(e_elec_csv_path, index=False)
    print(f"Saved E_elec metrics to: {e_elec_csv_path}")
    
    # Create ranking plots
    esp_ranking_path = os.path.join(args.output, "esp_method_ranking.png")
    esp_ranking_y_label = f"{KEY_LABEL_MAP[esp_charge_density]} RMSE" 
    plot_method_ranking(esp_metrics, esp_ranking_path, metric_name="RMSE", y_label=esp_ranking_y_label, unit=ESP_UNIT)
    
    e_elec_ranking_path = os.path.join(args.output, "e_elec_method_ranking.png")
    e_elec_ranking_y_label = f"{KEY_LABEL_MAP[elec_charge_density]} RMSE"
    plot_method_ranking(e_elec_metrics, e_elec_ranking_path, metric_name="RMSE", y_label=e_elec_ranking_y_label, unit=ENERGY_UNIT)
    
    print(f"\nAll plots saved to: {args.output}")

if __name__ == "__main__":
    main()
