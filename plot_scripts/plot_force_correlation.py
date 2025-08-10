#!/usr/bin/env python
# Intended use: comparison of forces from .xvgs from reruns and .extxyz files
# Expected units: .xvg files in kJ/mol/nm, .extxyz files in eV/Å
# Conversion to eV/Å is done internally 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pandas as pd
from typing import Dict, List, Optional, Tuple
from ase import Atoms
from ase.io import read

FORCES_KEY = "gromacs_force"

SCATTER_SAVENAME = "force_scatter_comparison.png"
TIMESERIES_SAVENAME = "force_max_difference.png"

FORCE_UNIT = "eV/Å"  # Unit for forces

DPI = 300 # DPI for saving plots

nm_TO_angstrom = 10.0  # Conversion factor from nm to Ångstrom
eV_TO_kJ_per_mol = 96.485  # Conversion factor from eV to kJ/mol
kJ_mol_nm_TO_eV_angstrom = (1/eV_TO_kJ_per_mol) / nm_TO_angstrom  # Conversion factor from kJ/mol/nm to eV/Å

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare and plot flattened force data from multiple files")
    parser.add_argument(
        "files",
        type=str,
        nargs="+",
        help="Paths to .xvg or .extxyz files (minimum 2 required). Reference Method first!"
    )
    parser.add_argument(
        "--labels", "-l",
        type=str,
        nargs="+",
        help="Labels for datasets (must match number of files)",
        default=None
    )
    parser.add_argument(
        "--fig-size",
        type=float,
        nargs=2,
        help="Figure size (width, height)",
        default=[10, 8]
    )
    parser.add_argument(
        "--alpha",
        type=float,
        help="Point transparency",
        default=0.5
    )
    parser.add_argument(
        "--point-size",
        type=float,
        help="Point size",
        default=10
    )
    args = parser.parse_args()
    if len(args.files) < 2:
        parser.error("At least two files must be provided.")
    if args.labels is not None and len(args.labels) != len(args.files):
        parser.error("Number of labels must match number of files.")
    if args.labels is None:
        args.labels = [f"Dataset {i+1}" for i in range(len(args.files))]
    return args

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

def plot_force_correlation(
    combined_df: pd.DataFrame,
    ref_label: str,
    stats: List[Dict[str, float]],
    args: argparse.Namespace,
) -> None:
    """
    Create and save a scatter plot comparing reference and method forces.

    Args:
        combined_df: DataFrame containing the data to plot.
        ref_label: Label for the reference method.
        stats: List of statistics dictionaries for each method.
        args: Parsed command-line arguments.
        UNIT: String representing the force unit.
        DPI: Dots per inch for saved figure.
    """
    sns.set_context("talk")
    fig, ax = plt.subplots(figsize=args.fig_size)

    #norm = plt.Normalize(combined_df["Step"].min(), combined_df["Step"].max())
    #sm = plt.cm.ScalarMappable(cmap="magma_r", norm=norm)

    sns.scatterplot(
        data=combined_df,
        x=f"{ref_label} Force",
        y="Method Force",
        #hue="Step",
        # hue_norm=norm,
        hue="Method",
        style="Method",
        palette="tab10",
        alpha=args.alpha,
        s=args.point_size,
        edgecolor=None,
        legend="brief",
        ax=ax
    )

    ax.set_xlabel(f"{ref_label} Force ({FORCE_UNIT})")
    ax.set_ylabel(f"Method Force ({FORCE_UNIT})")

    # cbar = plt.colorbar(sm, ax=ax)
    # cbar.set_label("Step")

    min_val = min(combined_df[f"{ref_label} Force"].min(), combined_df["Method Force"].min())
    max_val = max(combined_df[f"{ref_label} Force"].max(), combined_df["Method Force"].max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.7, linewidth=0.5)

    plt.legend(loc="lower right")
    for legend_handle in ax.get_legend().legend_handles:
        legend_handle.set_markersize(args.point_size)
        legend_handle.set_alpha(1.0)

    annotation_text = "Statistics:\n"
    for i, (label, stat) in enumerate(zip(args.labels[1:], stats)):
        annotation_text += f"{label}:\n"
        annotation_text += f"  RMSE: {stat['rmse']:.2f} {FORCE_UNIT}\n"
        annotation_text += f"  R²: {stat['r_squared']:.2f}\n"
        if i < len(stats) - 1:
            annotation_text += "\n"

    plt.annotate(
        annotation_text,
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        va="top"
    )

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(SCATTER_SAVENAME, dpi=DPI, bbox_inches='tight')
    plt.close()

def plot_max_force_difference(
        data: pd.DataFrame,
        ref_label: str,
        args: argparse.Namespace
) -> None:
    """
    Plot the maximum force difference for each method compared to the reference method.
    """
    sns.set_context("talk")
    fig, ax = plt.subplots(figsize=args.fig_size)

    # Calculate the maximum force difference for each method
    max_force_differences: pd.DataFrame = data.groupby(["Method", "Step"]).max("Abs. Force Difference").reset_index()
    #max_force_differences = max_force_differences.pivot(index="Step", columns="Method", values="Force Difference")

    # Plot the maximum force differences
    sns.lineplot(
        data=max_force_differences,
        x="Step",
        y="Abs. Force Difference",
        hue="Method",
        ax=ax,
        palette="tab10",
    )
    ax.set_xlabel("Step")
    ax.set_ylabel(r"Max. |$\Delta$ Force|" + f" [{FORCE_UNIT}]")

    # Calculate and mark mean of maximum force differences for each method
    mean_of_max_force_diff = max_force_differences.groupby("Method")["Abs. Force Difference"].mean()
    
    # Get the same colors as the lineplot (tab10 palette)
    palette = sns.color_palette("tab10", n_colors=len(mean_of_max_force_diff))
    
    # Add horizontal lines for mean values and y-ticks
    current_yticks = list(ax.get_yticks())
    current_yticklabels = [f"{tick:.2f}" for tick in current_yticks]
    
    for i, (method, mean_val) in enumerate(mean_of_max_force_diff.items()):
        # Add horizontal line across the entire plot
        ax.axhline(y=mean_val, color=palette[i], linestyle='--', alpha=0.7, linewidth=1)
        # Add the mean value to y-ticks
        current_yticks.append(mean_val)
        current_yticklabels.append(f"{mean_val:.3f}")
    
    # Update y-ticks to include mean values
    ax.set_yticks(current_yticks)
    ax.set_yticklabels(current_yticklabels)

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(TIMESERIES_SAVENAME, dpi=DPI, bbox_inches='tight')
    plt.close() 

def main() -> None:
    args = parse_args()

    # Read data files
    try:
        readouts: List[Tuple[Optional[np.ndarray], np.ndarray]] = [read_file(file) for file in args.files]
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return
    time_data = [readout[0] for readout in readouts]
    forces_data = [readout[1] for readout in readouts]
    for forces in forces_data:
        assert forces.shape == forces_data[0].shape, "All datasets must have the same shape."

    # Assert time matching if available
    first_valid_time = next((t for t in time_data if t is not None), None)
    if first_valid_time is not None:
        for time in time_data:
            if time is not None and not np.array_equal(time, first_valid_time):
                raise ValueError("Time vectors must match across all datasets.")

    ref_force = forces_data[0]  # Reference method is the first dataset
    ref_label = args.labels[0] if args.labels else "Reference"
    n_steps = ref_force.shape[0]
    n_entries_per_molecule = ref_force.shape[1] # flattened forces per molecule, n_atoms * 3
    steps = np.arange(n_steps).repeat(n_entries_per_molecule)

    flat_ref_force = ref_force.flatten()
    flat_method_forces = [forces.flatten() for forces in forces_data[1:]]  # Skip reference method

    # Calculate statistics
    stats = [calculate_statistics(flat_ref_force, flat_force) for flat_force in flat_method_forces]
    
    # Prepare dataframe for plotting
    dfs = []
    for label, flat_force in zip(args.labels[1:], flat_method_forces):
        df = pd.DataFrame({
            "Step": steps,
            "Time": np.repeat(first_valid_time, n_entries_per_molecule) if first_valid_time is not None else None,
            "Method Force": flat_force,
            f"{ref_label} Force": flat_ref_force,
            "Method": label
        })
        df["Abs. Force Difference"] = np.abs(df["Method Force"] - df[f"{ref_label} Force"])
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)

    plot_force_correlation(
        combined_df=combined_df,
        ref_label=ref_label,
        stats=stats,
        args=args,
    )
    plot_max_force_difference(
        data=combined_df,
        ref_label=ref_label,
        args=args
    )


if __name__ == '__main__':
    main()