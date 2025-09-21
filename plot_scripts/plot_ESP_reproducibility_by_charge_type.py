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

N_SUBSAMPLE = 10000
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
    "ESP_charges_loew": r"$\text{ESP}_\text{Loewdin}$",
    "ESP_charges_chelpg": r"$\text{ESP}_\text{CHELPG}$",
    "ESP_charges_mk": r"$\text{ESP}_\text{MK}$",
    "ESP_charges_resp": r"$\text{ESP}_\text{RESP}$",
    "E_elec_charge_density": r"$E_\text{elec, Charge density}$",
    "E_elec_charges_hirsh": r"$E_\text{elec, Hirshfeld}$",
    "E_elec_charges_mull": r"$E_\text{elec, Mulliken}$",
    "E_elec_charges_loew": r"$E_\text{elec, Loewdin}$",
    "E_elec_charges_chelpg": r"$E_\text{elec, CHELPG}$",
    "E_elec_charges_mk": r"$E_\text{elec, MK}$",
    "E_elec_charges_resp": r"$E_\text{elec, RESP}$",
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

def calculate_metrics(x: np.ndarray, y: np.ndarray) -> tuple:
    """Calculate RMSE and R² metrics."""
    rmse = np.sqrt(np.mean((x - y) ** 2))
    r2 = np.corrcoef(x, y)[0, 1] ** 2 if len(x) > 1 else 0.0
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
    ) -> None:

    n_charge_types = len(y_keys)
    n_cols = 2
    n_rows = (n_charge_types + n_cols - 1) // n_cols  # Round up to the nearest whole number

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * FIGSIZE[0], n_rows * FIGSIZE[1]))
    axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing

    # get 1st and 99th percentiles for axis limits
    # to avoid outliers affecting the axis limits too much
    low_quantile = data[[x_key]+y_keys].quantile(0.01).min()
    high_quantile = data[[x_key]+y_keys].quantile(0.99).max()

    for i, y_key in enumerate(y_keys):
        output_file = os.path.join(output_dir, f"{x_key}_vs_{y_key.replace('ESP_', '').replace('E_elec_', '')}.png")
        create_scatter_plot(
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
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    combined_output_file = os.path.join(output_dir, f"{x_key}_vs_all.png")
    plt.savefig(combined_output_file, dpi=DPI, bbox_inches='tight')
    plt.close()


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
    create_scatter_plots(
        data, 
        esp_charge_density, 
        esp_keys, 
        args.output,
        unit=ESP_UNIT
    )
    
    # Create plots comparing E_elec_charge_density to other E_elec methods
    e_elec_data = data.groupby('molecule_index').first().reset_index()
    elec_keys = [key for key in data.keys() if key.startswith('E_elec_') and key != elec_charge_density]
    create_scatter_plots(
        e_elec_data, 
        elec_charge_density, 
        elec_keys, 
        args.output,
        unit=ENERGY_UNIT
    )
    
    print(f"\nAll plots saved to: {args.output}")

if __name__ == "__main__":
    main()
