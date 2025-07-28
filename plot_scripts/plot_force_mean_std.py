#!/usr/bin/env python
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=analysis.out
#SBATCH --error=analysis.out

import argparse
import numpy as np
import os
import pandas as pd
import shutil
import sys
from typing import Tuple, List, Optional
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import MDAnalysis as mda
import seaborn as sns
from ase.data import atomic_numbers

# DEFAULT VALUES
MEAN_STD_FILE: str = "qm_mlmm_std.xyz" # Trajectory file at folder location, should only contain qm-atoms

N_LAST_TIMESTEPS: int = 0 # Number of last timesteps to plot, 0 for all
FIG_SIZE = (8,6)
DPI: int = 100 # DPI for saving plots

DEFAULT_COLLECTION_FOLDER_NAME: str = "std_analysis"

ENERGY_UNIT = "eV"
FORCE_UNIT = "eV/Å"

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze bond distances over time")
    ap.add_argument("-p", "--prefix", default=None, type=str, required=False, help="Prefix of directionaries with trajectories, default: None", metavar="prefix")
    ap.add_argument("-f", "--file", default=MEAN_STD_FILE, type=str, required=False, help=f"Trajectory file to analyze, default: {MEAN_STD_FILE}", metavar="trajectory_file")
    args = ap.parse_args()
    for key, value in vars(args).items():
        print(f"Argument {key}: {value}")
    return args
    
def validate_args(args: argparse.Namespace) -> None:
    if args.prefix is None:
        args.target_dir = os.getcwd()
        args.collection_folder_name = None
        args.present_dirs = ["."]  # Use relative path for current directory
    else:
        args.target_dir = os.path.dirname(os.path.abspath(args.prefix))

        args.collection_folder_name = os.path.join(args.target_dir, DEFAULT_COLLECTION_FOLDER_NAME)
        if not os.path.exists(args.collection_folder_name):
            os.makedirs(args.collection_folder_name)
        args.present_dirs = [present_dir for present_dir in os.listdir(args.target_dir) \
                             if os.path.isdir(os.path.join(args.target_dir, present_dir)) \
                             and present_dir.startswith(args.prefix)]
        
def main() -> None:
    args = parse_args()
    validate_args(args)

    valid_dirs = []
    for present_dir in args.present_dirs:
        if not os.path.exists(os.path.join(args.target_dir, present_dir, args.file)):
            continue
        valid_dirs.append(present_dir)
    if len(valid_dirs) > 1:
        try:
            valid_dirs = sorted(valid_dirs, key=lambda x: int(x.split("_")[-1]))  # Sort directories by the last part of their name
        except ValueError:
            print("Info: Unable to sort directories by numeric suffix, using original order.")
    print(f"Valid directories: {valid_dirs}")
    assert len(valid_dirs) > 0, "No valid directories found"

    root_dir = os.getcwd()
    walker_force_dfs: List[pd.DataFrame] = []
    for valid_dir in valid_dirs:
        print(f"Analyzing {valid_dir}")
        dir_path = os.path.join(args.target_dir, valid_dir)
        os.chdir(dir_path)
        walker_force_df: Optional[pd.DataFrame] = create_walker_force_df(args)
        os.chdir(root_dir)

        if walker_force_df is not None:
            # Use more descriptive naming for single directory case
            if args.prefix is None:
                walker_name = "current_dir"
            else:
                walker_name = valid_dir
            walker_force_df["Walker"] = walker_name
            walker_force_dfs.append(walker_force_df)
    
    if walker_force_dfs:
        walker_force_dfs = pd.concat(walker_force_dfs, ignore_index=True)
    else:
        print("No valid walker dataframes found, exiting.")
        sys.exit(0)
        
    if args.collection_folder_name is not None:
        os.chdir(args.collection_folder_name)

    sns.set_context("talk", font_scale=1.3)
    sns.set_style("whitegrid")
    sns.set_palette("tab10")
    print("Plotting largest force standard deviations")
    plot_largest_standard_deviations(walker_force_dfs)
    if len(valid_dirs) > 1:
        print("Plotting force standard deviations in subplots")
        plt_subplots(walker_force_dfs)

def create_walker_force_df(args: argparse.Namespace) -> pd.DataFrame:
    try:
        universe = mda.Universe(args.file)
    except Exception as e:
        print(e, file=sys.stderr)
        return None
    n_timesteps = len(universe.trajectory)//2 # Alternating mean and std in trajectory
    n_atoms = universe.atoms.n_atoms
    elements = universe.atoms.elements
    atomic_masses = universe.atoms.masses

    #Assumption: Mean and Stds alternate in trajectory
    means = np.array([atoms.positions.copy() for atoms in universe.trajectory[::2]]) # shape (n_timesteps, n_atoms, 3)
    stds = np.array([atoms.positions.copy() for atoms in universe.trajectory[1::2]]) # shape (n_timesteps, n_atoms, 3)
    weighted_stds = stds / atomic_masses[np.newaxis, :, np.newaxis]  # shape (n_timesteps, n_atoms, 3)

    timestep_indices = np.repeat(np.arange(n_timesteps), n_atoms*3)
    atom_indices = np.tile(np.repeat(np.arange(n_atoms), 3), n_timesteps)
    coordinate_indices = np.tile(np.array(["x", "y", "z"]), n_timesteps * n_atoms)
    element_indices = np.tile(np.repeat(elements, 3), n_timesteps)

    # Create a DataFrame for the bond distances
    data = {
        "Time Step": timestep_indices,
        "Atom Index": atom_indices,
        "Element": element_indices,
        "Coordinate": coordinate_indices,
        "Mean": means.flatten(),
        "Std": stds.flatten(),
        "Weighted Std": weighted_stds.flatten(),
    }

    walker_df = pd.DataFrame(data)

    # Apply time step filter if specified
    if N_LAST_TIMESTEPS > 0:
        n_last_timesteps_filter = walker_df["Time Step"] >= walker_df["Time Step"].nlargest(N_LAST_TIMESTEPS).min()
        walker_df = walker_df[n_last_timesteps_filter]

    return walker_df

def plot_largest_standard_deviations(walker_dfs: pd.DataFrame) -> None:
    walker_names = walker_dfs["Walker"].unique()
    for walker_name in walker_names:
        walker_df = walker_dfs[walker_dfs["Walker"] == walker_name]
        file_suffix = f"_{walker_name}" if walker_name != "current_dir" else ""
        
        max_std_per_timestep = walker_df.groupby("Time Step")["Std"].max().reset_index()
        print(max_std_per_timestep.head())

        fig, ax = plt.subplots(figsize=FIG_SIZE)
        plot_lineplot(max_std_per_timestep, ax=ax)

        plt.tight_layout()
        plt.savefig(f"force_std_max{file_suffix}.png", dpi=DPI)
        plt.close()

def plt_subplots(walker_dfs: pd.DataFrame) -> None:
    """
    Plot force stds maximum for each walker in subplots.
    Each subplot corresponds to a walker, showing the maximum force standard deviation over time.

    Parameters:
    walker_dfs (pd.DataFrame): DataFrame containing the force standard deviations for each walker
    """
    walker_labels = walker_dfs["Walker"].unique()
    try:
        walker_labels = sorted(walker_labels, key=lambda x: int(x.split("_")[-1]))  # Sort by the last part of the walker label
    except ValueError:
        print("Info: Unable to sort walker labels by numeric suffix, using original order.")
    n_walkers = len(walker_labels)
    
    # Calculate subplot grid dimensions
    n_cols = min(3, n_walkers)  # Maximum 3 columns
    n_rows = (n_walkers + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    sns.set_context("talk", font_scale=1.0)
    
    # Flatten axes array for easier indexing
    if n_walkers == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    # Plot for each walker
    for idx, walker_label in enumerate(walker_labels):
        walker_df: pd.DataFrame = walker_dfs[walker_dfs["Walker"] == walker_label]
        max_std_per_timestep: pd.DataFrame = walker_df.groupby("Time Step")["Std"].max().reset_index()

        # Plot in the corresponding subplot
        ax = axes[idx]
        plot_lineplot(max_std_per_timestep, ax=ax)
        
        ax.set_title(f"{walker_label}")
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # Hide unused subplots
    for idx in range(n_walkers, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig("force_std_max_all_walkers.png", dpi=DPI, bbox_inches='tight')
    plt.close()

def plot_lineplot(data: pd.DataFrame, ax=None) -> None:
    sns.lineplot(
        data=data, 
        x="Time Step", 
        y="Std",
        ax=ax,
    )
    ax.set_xlabel("Time Step")
    ax.set_ylabel(f"Max. Force Std Dev [{FORCE_UNIT}]")
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    

if __name__ == "__main__":
    main()