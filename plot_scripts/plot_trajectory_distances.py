#!/usr/bin/env python
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=dist_analysis.out
#SBATCH --error=dist_analysis.out

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
import MDAnalysis as mda
from MDAnalysis.topology.guessers import guess_atom_element
import seaborn as sns
from ase.data import atomic_numbers

# DEFAULT VALUES
TRAJECTORY_FILE: str = "qm.xtc" # Trajectory file at folder location, should only contain qm-atoms
TOPOLOGY_FILE: str = "qm_topol.top" # Path to the topology file (e.g., .gro or .pdb), should be the same for each trajectory
#Note: Gromacs intern itp library may need to be made accessible by something like
# ln -s /usr/local/run/gromacs-dftbplus-machine-learning/share/gromacs/top/amber99sb-ildn.ff/

N_PLOTS: int = 5 # Number of plots to generate, chosen from the bonds with the largest, smallest values and standard deviation
N_LAST_TIMESTEPS: int = 0 # Number of last timesteps to plot, 0 for all
DPI: int = 100 # DPI for saving plots

DEFAULT_COLLECTION_FOLDER_NAME: str = "bond_distance_analysis"
AMBER_ILDN_PATH: str = "/lustre/home/ka/ka_ipc/ka_he8978/gromacs-orca/share/gromacs/top/amber99sb-ildn.ff"

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze bond distances over time")
    ap.add_argument("-p", "--prefix", default=None, type=str, required=False, help="Prefix of directionaries with trajectories, default: None", metavar="prefix")
    ap.add_argument("-t", "--trajectory_file", default=TRAJECTORY_FILE, type=str, required=False, help="Relative path to the trajectory file", metavar="trajectory_file")
    ap.add_argument("-top", "--topology_file", default=TOPOLOGY_FILE, type=str, required=False, help="Relative path to the topology file", metavar="topology_file")
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
    
    args.topology_file = os.path.join(args.target_dir, args.topology_file)

    if not os.path.exists(args.topology_file):
        raise FileNotFoundError(f"Topology file '{args.topology_file}' does not exist.")
        
def main() -> None:
    args = parse_args()
    validate_args(args)

    # Ensure the amber99sb-ildn.ff is accessible
    topology_folder = os.path.dirname(args.topology_file)
    if not os.path.exists(os.path.join(topology_folder, "amber99sb-ildn.ff")):
        os.symlink(AMBER_ILDN_PATH, os.path.join(topology_folder, "amber99sb-ildn.ff"), target_is_directory=True) 

    valid_dirs = []
    for present_dir in args.present_dirs:
        if not os.path.exists(os.path.join(args.target_dir, present_dir, args.trajectory_file)):
            print(f"Skipping {present_dir}: {args.trajectory_file} does not exist.")
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
    walker_bonds_dfs: List[pd.DataFrame] = []
    for valid_dir in valid_dirs:
        print(f"Analyzing {valid_dir}")
        dir_path = os.path.join(args.target_dir, valid_dir)
        os.chdir(dir_path)
        walker_bonds_df: Optional[pd.DataFrame] = create_walker_bonds_df(args)
        os.chdir(root_dir)

        if walker_bonds_df is not None:
            # Use more descriptive naming for single directory case
            if args.prefix is None:
                walker_name = "current_dir"
            else:
                walker_name = valid_dir
            walker_bonds_df["Walker"] = walker_name
            walker_bonds_dfs.append(walker_bonds_df)
    
    if walker_bonds_dfs:
        walker_bonds_dfs = pd.concat(walker_bonds_dfs, ignore_index=True)
    else:
        print("No valid walker dataframes found, exiting.")
        sys.exit(0)
        
    if args.collection_folder_name is not None:
        os.chdir(args.collection_folder_name)

    print("Analyzing extreme bond distances...")
    plot_extreme_bond_distances(walker_bonds_dfs)
    if len(valid_dirs) > 1:
        print("Plotting bond distances in subplots...")
        plt_subplots(walker_bonds_dfs)
    print("Analyzing global bond distances...")
    plot_bond_length_distribution(walker_bonds_dfs)
    print("Analyzing hydrogen bond lengths...")
    plot_h_bond_length_distribution(walker_bonds_dfs)
    print("Analysis complete")

def create_walker_bonds_df(args: argparse.Namespace) -> pd.DataFrame:
    # Suppress MDAnalysis deprecation warnings
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        module="MDAnalysis.topology.ITPParser"
    )

    try:
        if args.topology_file.endswith(".top"):
            universe = mda.Universe(args.topology_file, args.trajectory_file, topology_format="itp")
        else:
            universe = mda.Universe(args.topology_file, args.trajectory_file)
    except Exception as e:
        print(f"{os.getcwd()}: Problem with Topology, skipping analysis", file=sys.stderr)
        print(e, file=sys.stderr)
        return None
    
    qm_atoms = universe.atoms
    try:
        qm_atoms.unwrap()  # Unwrap the atoms to avoid periodic boundary issues
    except Exception as e:
        print("Warning: Unwrapping atoms failed, continuing without unwrapping.", file=sys.stderr)
    unique_edge_indices, elements_bonds, atomic_numbers_bonds = get_atomic_numbers_and_elements(qm_atoms)
    bond_labels = [f"{elements_bond[0]}{unique_edge_indices[i, 0]}-{elements_bond[1]}{unique_edge_indices[i, 1]}" for i, elements_bond in enumerate(elements_bonds)]
    n_timesteps = len(universe.trajectory)
    n_bonds = len(unique_edge_indices)

    bond_distances_all_timesteps = []
    times = []  # Store times in picoseconds
    for timestep in universe.trajectory:
        # Calculate the distance matrix
        bond_distances = mda.lib.distances.calc_bonds(qm_atoms.positions[unique_edge_indices[:, 0]],
                                                           qm_atoms.positions[unique_edge_indices[:, 1]],
                                                           box=universe.dimensions) # shape: (n_bonds,
        bond_distances_all_timesteps.append(bond_distances)
        times.append(timestep.time)  # Time in ps
    
    bond_distances_all_timesteps = np.stack(bond_distances_all_timesteps, axis=0)  # shape: (n_timesteps, n_bonds)
    bond_distance_maxs = bond_distances_all_timesteps.max(axis=0) # shape: (n_bonds,)
    bond_distance_stds = bond_distances_all_timesteps.std(axis=0) # shape: (n_bonds,)

    # Convert times from ps to ns 
    #times = np.array(times) / 1e4 

    time_values = np.repeat(times, n_bonds)  # Repeat each time value n_bonds times, shape: (n_timesteps * n_bonds,)
    timestep_indices = np.repeat(np.arange(n_timesteps), n_bonds) # Repeat each entry n times, shape: (n_timesteps * n_bonds,)
    bond_indices = np.tile(np.arange(n_bonds), n_timesteps) # Repeat the array n times, shape: (n_timesteps * n_bonds,)
    bond_labels = np.tile(bond_labels, n_timesteps) # shape: (n_timesteps * n_bonds,)
    elements_bond_partner1 = np.tile(elements_bonds[:, 0], n_timesteps) # shape: (n_timesteps * n_bonds,)
    elements_bond_partner2 = np.tile(elements_bonds[:, 1], n_timesteps) # shape: (n_timesteps * n_bonds,)
    atomic_numbers_bond_partner1 = np.tile(atomic_numbers_bonds[:, 0], n_timesteps) # shape: (n_timesteps * n_bonds,)
    atomic_numbers_bond_partner2 = np.tile(atomic_numbers_bonds[:, 1], n_timesteps) # shape: (n_timesteps * n_bonds,)

    # Create a DataFrame for the bond distances
    data = {
        "Time (ps)": time_values,
        "Time Step": timestep_indices,
        "Bond Index": bond_indices,
        "Bond Label": bond_labels,
        "Element 1": elements_bond_partner1,
        "Element 2": elements_bond_partner2,
        "Atomic Number 1": atomic_numbers_bond_partner1,
        "Atomic Number 2": atomic_numbers_bond_partner2,
        "Bond Distance": bond_distances_all_timesteps.flatten(),
    }

    walker_df = pd.DataFrame(data)

    return walker_df

def plot_extreme_bond_distances(
    walker_dfs: pd.DataFrame
) -> None:
    """
    Plot the highest/lowest bond distances and bonds with the highest/lowest standard deviations.

    Args:
        walker_df (pd.DataFrame): DataFrame containing bond distance data.
    """
    walker_labels = sorted(walker_dfs["Walker"].unique())
    for walker_label in walker_labels:
        walker_df = walker_dfs[walker_dfs["Walker"] == walker_label]
        file_suffix = f"_{walker_label}" if walker_label != "current_dir" else ""

        # Get the bond distance max and std
        bond_distance_all = walker_df.groupby("Bond Label")["Bond Distance"].max().sort_values(ascending=False)
        bond_distance_maxs = walker_df.groupby("Bond Label")["Bond Distance"].max().nlargest(N_PLOTS).sort_values(ascending=False)
        bond_distance_mins = walker_df.groupby("Bond Label")["Bond Distance"].min().nsmallest(N_PLOTS).sort_values(ascending=False)
        bond_distance_max_stds = walker_df.groupby("Bond Label")["Bond Distance"].std().nlargest(N_PLOTS).sort_values(ascending=False)
        bond_distance_min_stds = walker_df.groupby("Bond Label")["Bond Distance"].std().nsmallest(N_PLOTS).sort_values(ascending=False)

        bond_labels = bond_distance_all.index.to_list()
        max_bond_labels = bond_distance_maxs.index.to_list()
        min_bond_labels = bond_distance_mins.index.to_list()
        max_std_bond_labels = bond_distance_max_stds.index.to_list()
        min_std_bond_labels = bond_distance_min_stds.index.to_list()

        max_df = walker_df[walker_df["Bond Label"].isin(max_bond_labels)].copy()
        min_df = walker_df[walker_df["Bond Label"].isin(min_bond_labels)].copy()
        max_std_df = walker_df[walker_df["Bond Label"].isin(max_std_bond_labels)].copy()
        min_std_df = walker_df[walker_df["Bond Label"].isin(min_std_bond_labels)].copy()

        walker_df["Bond Label"] = pd.Categorical(walker_df["Bond Label"], categories=bond_labels, ordered=True)
        max_df["Bond Label"] = pd.Categorical(max_df["Bond Label"], categories=max_bond_labels, ordered=True)
        min_df["Bond Label"] = pd.Categorical(min_df["Bond Label"], categories=min_bond_labels, ordered=True)
        max_std_df["Bond Label"] = pd.Categorical(max_std_df["Bond Label"], categories=max_std_bond_labels, ordered=True)
        min_std_df["Bond Label"] = pd.Categorical(min_std_df["Bond Label"], categories=min_std_bond_labels, ordered=True)

        walker_df = walker_df.sort_values(by=["Bond Label", "Time Step"], ascending=[False, True])
        max_df = max_df.sort_values(by=["Bond Label", "Time Step"], ascending=[False, True])
        min_df = min_df.sort_values(by=["Bond Label", "Time Step"], ascending=[False, True])
        max_std_df = max_std_df.sort_values(by=["Bond Label", "Time Step"], ascending=[False, True])
        min_std_df = min_std_df.sort_values(by=["Bond Label", "Time Step"], ascending=[False, True])

        plot_bond_distances(walker_df, title=f"bond_distance_all{file_suffix}.png")
        plot_bond_distances(max_df, title=f"bond_distance_max{file_suffix}.png")
        plot_bond_distances(min_df, title=f"bond_distance_min{file_suffix}.png")
        plot_bond_distances(max_std_df, title=f"bond_distance_max_std{file_suffix}.png")
        plot_bond_distances(min_std_df, title=f"bond_distance_min_std{file_suffix}.png")

def get_atomic_numbers_and_elements(atoms: mda.AtomGroup) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the atomic numbers and elements of the atoms in the bonds.

    Args:
        atoms (mda.AtomGroup): The atoms in the system.

    Returns:
        Tuple[inp.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing the following:
            - unique_edge_indices (np.ndarray): An array of unique edge indices.
            - elements_bonds (np.ndarray): An array of elements of the atoms in the bonds. Guesses the element symbol from the atom name.
            - atomic_numbers_bonds (np.ndarray): An array of atomic numbers of the atoms in the bonds. Depends on the element symbol guess.
    """
    unique_edge_indices = []
    elements_bonds = []
    atomic_numbers_bonds = []
    for atom_a in atoms:
        for atom_b in atom_a.bonded_atoms:
            if atom_a.index < atom_b.index:
                unique_index = [atom_a.index, atom_b.index]
                try:
                    # Use the actual element if available
                    elements_bond = [atom_a.element, atom_b.element]
                except mda.exceptions.NoDataError:
                    # Fallback to the element guess
                    elements_bond = [guess_atom_element(atom_a.name), guess_atom_element(atom_b.name)]
                except Exception as e:
                    print(f"Error guessing element for atoms {atom_a.index} and {atom_b.index}: {e}")
                    raise
                atomic_numbers_bond = [atomic_numbers.get(element, 0) for element in elements_bond]
                unique_edge_indices.append(unique_index)
                elements_bonds.append(elements_bond)
                atomic_numbers_bonds.append(atomic_numbers_bond)

    unique_edge_indices = np.array(unique_edge_indices) # shape: (n_bonds, 2)
    elements_bonds = np.array(elements_bonds) # shape: (n_bonds, 2)
    atomic_numbers_bonds = np.array(atomic_numbers_bonds) # shape: (n_bonds, 2)

    return unique_edge_indices, elements_bonds, atomic_numbers_bonds

def plot_bond_distances(bond_distances_df: pd.DataFrame, title: str = "bond_distances_vs_time_steps.png"):
    # Plot the bond distances over time
    plt.figure(figsize=(10,10))
    sns.set_context(context="talk", font_scale=1.3)
    if N_LAST_TIMESTEPS > 0:
        n_last_timesteps_filter = bond_distances_df["Time Step"] >= bond_distances_df["Time Step"].nlargest(N_LAST_TIMESTEPS).min()
        bond_distances_df = bond_distances_df[n_last_timesteps_filter]

    sns.lineplot(data=bond_distances_df, palette="tab10", x="Time (ps)", y="Bond Distance", hue="Bond Label")
    plt.xlabel("Time (ps)")
    plt.ylabel("Bond Distance [Å]")
    plt.legend(title="Bonds", bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig(title, dpi=DPI)
    plt.close()

def plot_h_bond_length_distribution(walker_dfs: pd.DataFrame, title: str = "hydrogen_bond_length_distribution.png"):
    # Get the bonds involving hydrogen atoms
    is_h_bond_involved = (walker_dfs["Element 1"] == "H") | (walker_dfs["Element 2"] == "H")
    h_bond_df = walker_dfs[is_h_bond_involved]
    
    if h_bond_df.empty:
        print("No hydrogen bonds found, skipping hydrogen bond distribution plot.")
        return
        
    binrange = (h_bond_df["Bond Distance"].min(), min(h_bond_df["Bond Distance"].max(), 1.3))  # Limit the range to 1.3 Å

    # Plot the distribution of hydrogen bond lengths
    plt.figure(figsize=(10,10))
    sns.set_context(context="talk", font_scale=1.3)
    sns.histplot(h_bond_df,
                 x="Bond Distance", 
                 hue="Bond Label", 
                 palette="tab10", 
                 multiple="stack", 
                 stat="probability", 
                 common_norm=True,
                 binrange=binrange,
                 bins=100)
    plt.xlabel("Hydrogen Bond Length [Å]")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(title, dpi=DPI)
    plt.close()

def plot_bond_length_distribution(walker_dfs: pd.DataFrame, title: str = "bond_length_distribution.png"):
    binrange = (walker_dfs["Bond Distance"].min(), min(walker_dfs["Bond Distance"].max(), 2.5))  # Limit the range to 2.5 Å
    plt.figure(figsize=(10,10))
    sns.set_context(context="talk", font_scale=1.3)
    sns.histplot(walker_dfs, 
                 x="Bond Distance", 
                 hue="Bond Label", 
                 palette="tab10", 
                 multiple="stack", 
                 stat="probability", 
                 common_norm=True,
                 binrange=binrange,
                 bins=100)
    plt.xlabel("Bond Length [Å]")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(title, dpi=DPI)
    plt.close()

def plt_subplots(walker_dfs: pd.DataFrame) -> None:
    """
    Plot bond distances for all walkers in subplots.
    
    Args:
        walker_dfs (pd.DataFrame): DataFrame containing bond distance data for all walkers.
    """
    walker_labels = sorted(walker_dfs["Walker"].unique(), key=lambda x: int(x.split("_")[-1]))  # Sort by the last part of the walker label
    n_walkers = len(walker_labels)
    
    # Calculate subplot grid dimensions
    n_cols = min(4, n_walkers)  # Maximum 4 columns
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
        walker_df = walker_dfs[walker_dfs["Walker"] == walker_label]
        
        
        # Apply time step filter if specified
        if N_LAST_TIMESTEPS > 0:
            n_last_timesteps_filter = walker_df["Time Step"] >= walker_df["Time Step"].nlargest(N_LAST_TIMESTEPS).min()
            walker_df = walker_df[n_last_timesteps_filter]
        
        # Plot in the corresponding subplot
        ax = axes[idx]
        sns.lineplot(
            data=walker_df, 
            x="Time (ps)", 
            y="Bond Distance", 
            hue="Bond Label",
            palette="tab10",
            ax=ax
        )
        
        ax.set_title(f"{walker_label}")
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel("Bond Distance [Å]")
        # remove the legend for subplots
        ax.get_legend().remove()

    # Hide unused subplots
    for idx in range(n_walkers, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig("bond_distance_all_walkers.png", dpi=DPI, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()