#!/usr/bin/env python
# Intended use: comparison of forces from .xvgs from reruns and .extxyz files 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pandas as pd
from typing import Dict, List, Optional, Tuple
from ase import Atoms
from ase.io import read

FORCES_KEY = "gromacs_force"

UNIT = "eV/Å"  # Unit for forces

nm_TO_angstrom = 10.0  # Conversion factor from nm to Ångstrom
eV_TO_kJ_per_mol = 96.485  # Conversion factor from eV to kJ/mol
kJ_mol_nm_TO_eV_angstrom = (1/eV_TO_kJ_per_mol) / nm_TO_angstrom  # Conversion factor from kJ/mol/nm to eV/Å

def read_file(file_path: str) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Read a .xvg or .extxyz file and return the data as a numpy array.

    Args:
        file_path: Path to the file to be read.
    Returns:
        Tuple containing:
            - time: NumPy array of time data (only for .xvg files).
            - forces: NumPy array of forces data.
    """
    if file_path.endswith('.xvg'):
        data = read_xvg_file(file_path)
        time, forces = data[:, 0], data[:, 1:]
        forces = forces * kJ_mol_nm_TO_eV_angstrom  # Convert forces from kJ/mol/nm to eV/Å
        return time, forces
    elif file_path.endswith('.extxyz'):
        forces = read_extxyz_file(file_path)
        return None, forces  # No time data in EXTXYZ, only forces
    else:
        raise ValueError(f"Unsupported file format: {file_path}. Only .xvg and .extxyz files are supported.")


def read_xvg_file(file_path: str) -> np.ndarray:
    """
    Read XVG file and return data as numpy array using np.loadtxt.
    
    Args:
        file_path: Path to the XVG file
        
    Returns:
        NumPy array containing the data
    """
    # Determine the number of comment and metadata lines
    skip_rows = 0
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith(('#', '@')):
                skip_rows += 1
            else:
                break
    
    # Load the data using np.loadtxt
    try:
        data = np.loadtxt(file_path, skiprows=skip_rows)
    except ValueError as e:
        raise ValueError(f"Error reading data from {file_path}: {e}")
    
    if data.size == 0:
        raise ValueError(f"No valid data found in {file_path}")
        
    return data

def read_extxyz_file(file_path: str) -> np.ndarray:
    """
    Read EXTXYZ file and return data as numpy array.
    
    Args:
        file_path: Path to the EXTXYZ file
    Returns:
        NumPy array containing the data
    """
    try:
        atoms: List[Atoms] = read(file_path, format='extxyz', index=':')
    except Exception as e:
        raise ValueError(f"Error reading EXTXYZ file {file_path}: {e}")
    
    if not atoms:
        raise ValueError(f"No valid data found in {file_path}")
    
    # Extract forces from the Atoms objects
    forces = []
    for atom in atoms:
            forces.append(atom.arrays[FORCES_KEY].flatten())
    forces = np.array(forces)

    return forces


def calculate_statistics(data1: np.ndarray, data2: np.ndarray) -> Dict[str, float]:
    """
    Calculate statistical measures between two datasets.
    
    Args:
        data1: First dataset
        data2: Second dataset
        
    Returns:
        Dictionary containing correlation coefficient, RMSE, and R²
    """
    corr = np.corrcoef(data1, data2)[0, 1]
    rmse = np.sqrt(np.mean((data1 - data2)**2))
    # R² calculation
    ss_tot = np.sum((data2 - np.mean(data2))**2)
    ss_res = np.sum((data2 - data1)**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return {
        'correlation': corr,
        'rmse': rmse,
        'r_squared': r_squared
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare and plot flattened XVG data')
    parser.add_argument('file1', type=str, help='Path to first XVG file')
    parser.add_argument('file2', type=str, help='Path to second XVG file')
    parser.add_argument('--output', '-o', type=str, help='Output file path', default='xvg_comparison.png')
    parser.add_argument('--title', '-t', type=str, help='Plot title', default='')
    parser.add_argument('--labels', '-l', type=str, nargs=2, help='Labels for datasets', 
                        default=['Forces 1', 'Forces 2'])
    parser.add_argument('--fig-size', type=float, nargs=2, help='Figure size (width, height)', 
                        default=[10, 8])
    parser.add_argument('--alpha', type=float, help='Point transparency', default=0.5)
    parser.add_argument('--point-size', type=float, help='Point size', default=10)
    args = parser.parse_args()

    # Read data files
    try:
        time1, forces1 = read_file(args.file1)
        time2, forces2 = read_file(args.file2)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return

    assert forces1.shape == forces2.shape, "Both datasets must have the same shape."



    # Assert time matching if available
    if time1 is not None and time2 is not None:
        if len(time1) != len(time2):
            raise ValueError("Time vectors must have the same length for both datasets.")
        assert np.array_equal(time1, time2), "Time vectors must match for both datasets."

    n_steps = forces1.shape[0]
    n_entries_per_molecule = forces1.shape[1]
    steps = np.arange(n_steps).repeat(n_entries_per_molecule)

    flat_forces1 = forces1.flatten()
    flat_forces2 = forces2.flatten()

    # Calculate statistics
    stats = calculate_statistics(flat_forces1, flat_forces2)
    
    # Prepare dataframe for plotting
    df = pd.DataFrame({
        args.labels[0]: flat_forces1,
        args.labels[1]: flat_forces2,
        "Step": steps,
        "Expected false": steps>=13
    })

    true_df = df[df["Expected false"] == False]
    false_df = df[df["Expected false"] == True]
    
    # Create scatter plot with color scale for time
    sns.set_context("talk")
    plt.figure(figsize=tuple(args.fig_size))
    
    # Add color bar for steps
    norm = plt.Normalize(df["Step"].min(), df["Step"].max())
    sm = plt.cm.ScalarMappable(cmap="magma_r", norm=norm)
    
    scatter_true = plt.scatter(
        x=true_df[args.labels[0]], 
        y=true_df[args.labels[1]], 
        c=true_df["Step"], 
        cmap=sm.cmap, 
        norm=sm.norm,
        alpha=args.alpha, 
        s=args.point_size,
        marker='o',
        label='Assumed Correct'
    )

    scatter_false = plt.scatter(
        x=false_df[args.labels[0]], 
        y=false_df[args.labels[1]], 
        c=false_df["Step"], 
        cmap=sm.cmap,
        norm=sm.norm,
        alpha=args.alpha, 
        s=args.point_size,
        marker='X',
        label='Assumed Incorrect'
    )
    
    plt.xlabel(f"{args.labels[0]} ({UNIT})")
    plt.ylabel(f"{args.labels[1]} ({UNIT})")

    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label("Step")
    
    # Add perfect correlation line
    min_val = min(df[args.labels[0]].min(), df[args.labels[1]].min())
    max_val = max(df[args.labels[0]].max(), df[args.labels[1]].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
    
    plt.legend(loc='lower right')
    for legend_handle in plt.gca().get_legend().legend_handles:
        legend_handle.set_sizes([args.point_size])
        legend_handle.set_alpha(1.0)  # Set legend point transparency to 1.0
        legend_handle.set_color('black')  # Set legend point color to black

    # Add annotation with statistics
    annotation_text = (
        f'Correlation: {stats["correlation"]:.4f}\n'
        f'RMSE: {stats["rmse"]:.4f}\n'
        f'R²: {stats["r_squared"]:.4f}\n'
        f'n = {n_steps} Steps'
    )
    
    plt.annotate(annotation_text, 
                xy=(0.05, 0.95), 
                xycoords='axes fraction', 
                fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
                va='top')
    
    plt.title(args.title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(args.output, dpi=300)
    print(f"Plot saved to {args.output}")
    
    plt.show()


if __name__ == '__main__':
    main()