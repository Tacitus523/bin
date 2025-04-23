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
from typing import Tuple
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import MDAnalysis as mda
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

default_collection_folder_name: str = "bond_distance_analysis"

def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze bond distances over time")
    ap.add_argument("-p", "--prefix", default=None, type=str, dest="prefix", action="store", required=False, help="Prefix of directionaries with trajectories, default: None", metavar="prefix")
    ap.add_argument("-t", "--trajectory_file", default=TRAJECTORY_FILE, type=str, dest="trajectory_file", action="store", required=False, help="Relative path to the trajectory file", metavar="trajectory_file")
    ap.add_argument("-top", "--topology_file", default=TOPOLOGY_FILE, type=str, dest="topology_file", action="store", required=False, help="Relative path to the topology file, not a .top, but a .gro or similar", metavar="topology_file")
    args = ap.parse_args()
    if args.prefix is None:
        present_dirs = [os.getcwd()]
        collection_folder_name = None
    else:
        target_dir = os.path.dirname(args.prefix)
        if target_dir == "":
            target_dir = "."
        os.chdir(target_dir)
        prefix = os.path.basename(args.prefix)
        present_dirs = [dir for dir in os.listdir(".") if os.path.isdir(dir) and dir.startswith(prefix)]
        collection_folder_name = os.path.abspath(default_collection_folder_name)
        if not os.path.exists(collection_folder_name):
            os.makedirs(collection_folder_name)
    args.topology_file = os.path.abspath(args.topology_file)

    valid_dirs = []
    for dir in present_dirs:
        if not os.path.exists(os.path.join(dir, args.trajectory_file)):
            continue
        valid_dirs.append(dir)
    valid_dirs = sorted(valid_dirs)
    print(f"Valid directories: {valid_dirs}")
    assert len(valid_dirs) > 0, "No valid directories found"

    root_dir = os.getcwd()
    bond_distances_all_walkers = []
    for dir in valid_dirs:
        print(f"Analyzing {dir}")
        os.chdir(dir)
        local_bond_distances_result = analyze_local_bond_distances(args, collection_folder_name=collection_folder_name)
        os.chdir(root_dir)
        
        if local_bond_distances_result is not None:
            bond_distances, edge_indices, atomic_numbers_bonds, elements_bonds = local_bond_distances_result
            bond_distances_all_walkers.append(bond_distances)
    
    bond_distances_all_walkers = np.concatenate(bond_distances_all_walkers, axis=0) # shape: (n_timesteps_all_walkers, n_atoms, n_atoms)
    # edge_indices, atomic_numbers_bonds, elements_bonds are expected to be the same for all walkers

    if collection_folder_name is not None:
        os.chdir(collection_folder_name)

    analyze_global_bond_distances(bond_distances_all_walkers, edge_indices, atomic_numbers_bonds, elements_bonds) 


def analyze_local_bond_distances(args: argparse.Namespace, collection_folder_name: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    try:
        universe = mda.Universe(args.topology_file, args.trajectory_file, topology_format="itp")
    except Exception as e:
        print(f"{os.getcwd()}: Problem with Topology, skipping analysis", file=sys.stderr)
        print(e, file=sys.stderr)
        return None

    qm_atoms = universe.atoms
    unique_edge_indices, elements_bonds, atomic_numbers_bonds = get_atomic_numbers_and_elements(qm_atoms)

    distance_matrices = []
    for timestep in universe.trajectory:
        # Calculate the distance matrix
        distance_matrix = mda.lib.distances.distance_array(qm_atoms.positions, qm_atoms.positions)
        distance_matrices.append(distance_matrix)
    
    distance_matrices = np.stack(distance_matrices, axis=0) # shape: (n_timesteps, n_atoms, n_atoms)
    bond_distances_all_timesteps = distance_matrices[:, unique_edge_indices[:, 0], unique_edge_indices[:, 1]] # shape: (n_timesteps, n_bonds)
    bond_distance_maxs = bond_distances_all_timesteps.max(axis=0)
    bond_distance_stds = bond_distances_all_timesteps.std(axis=0)

    # plot highest/lowest bond distances and bonds with highest/lowest standard deviations
    for data_label, bond_distance_data in zip(["bond_distances", "bond_distance_stds"],[bond_distance_maxs, bond_distance_stds]):
        size_labels = []
        indices_list = []
        if data_label == "bond_distances":
            sorted_indices = np.argsort(bond_distance_data)[::-1] # Sort the bonds by the maximum bond distance
            size_labels.append("all")
            indices_list.append(sorted_indices)
        largest_bond_distance_data_indices = np.argsort(bond_distance_data)[-N_PLOTS:][::-1] # Only plot the N_PLOTS bonds with the largest values
        smallest_bond_distance_data_indices = np.argsort(bond_distance_data)[:N_PLOTS][::-1] # Only plot the N_PLOTS bonds with the smallest values
        size_labels.extend(["largest", "smallest"])
        indices_list.extend([largest_bond_distance_data_indices, smallest_bond_distance_data_indices])
        for size_label, indices in zip(size_labels, indices_list):
            extreme_bond_data = bond_distances_all_timesteps[:, indices]

            target_edge_indices = unique_edge_indices[indices] # Get the edge indices of the bonds with the smallest/largest bond distances or all bonds
            target_atomic_numbers_bonds = atomic_numbers_bonds[indices] # Get the atomic numbers of the atoms in the bonds with the smallest/largest bond distances or all bonds
            target_element_bonds = elements_bonds[indices] # Get the elements of the atoms in the bonds with the smallest/largest bond distances or all bonds

            labels = [f"({target_element_bonds[i, 0]}{target_edge_indices[i, 0]}, {target_element_bonds[i, 1]}{target_edge_indices[i, 1]})" for i in range(len(target_edge_indices))]
            bond_distances_df = pd.DataFrame(extreme_bond_data, columns=labels)
            bond_distances_df["Time Step"] = np.arange(len(universe.trajectory))
            bond_distances_df = bond_distances_df.set_index("Time Step")
            title = f"{data_label}_{size_label}_vs_time_steps.png"
            plot_bond_distances(bond_distances_df, title=title)
    
            if collection_folder_name is not None:
                folder_name = os.path.basename(os.getcwd())
                collection_title = f"{data_label}_{size_label}_vs_time_steps_{folder_name}.png"
                shutil.copy(title, os.path.join(collection_folder_name, collection_title))
    
    return bond_distances_all_timesteps, unique_edge_indices, atomic_numbers_bonds, elements_bonds

def analyze_global_bond_distances(bond_distances: np.ndarray, edge_indices: np.ndarray, atomic_numbers_bonds: np.ndarray, elements_bonds: np.ndarray) -> None:
    bond_types = [f"({elements_bond[0]}-{elements_bond[1]})" for elements_bond in elements_bonds]
    flat_bond_distances = bond_distances.flatten()
    bond_distances_df = pd.DataFrame(flat_bond_distances, columns=["Bond Length"])
    bond_distances_df["Bond Type"] = bond_types*np.shape(bond_distances)[0] # Repeat the bond types for each timestep
    plot_bond_length_distribution(bond_distances_df)

    # Get the bonds involving hydrogen atoms
    is_h_bond_involved = np.any(atomic_numbers_bonds == 1, axis=1) # Check if any of the atoms in the bond is a hydrogen atom, shape: (n_bonds,)
    h_bond_edges = edge_indices[is_h_bond_involved] # Get the edge indices of the bonds involving hydrogen atoms, shape: (n_h_bonds, 2)
    
    # Get the bond lengths of the bonds involving hydrogen atoms
    h_bond_lengths = bond_distances[:, is_h_bond_involved] # shape: (n_timesteps, n_h_bonds,)
    h_bond_lengths = h_bond_lengths.flatten() # shape: (n_timesteps * n_h_bonds,)
    bond_types = [
        f"({elements_bond[0]}-{elements_bond[1]})" if elements_bond[1] == "H" 
        else f"({elements_bond[1]}-{elements_bond[0]})"
        for elements_bond in elements_bonds[is_h_bond_involved]
    ] # Get the bond types of the bonds involving hydrogen atoms, shape: (n_h_bonds,)

    h_bond_df = pd.DataFrame(h_bond_lengths, columns=["Hydrogen Bond Length"])
    h_bond_df["Bond Type"] = bond_types*np.shape(bond_distances)[0] # Repeat the bond types for each timestep

    plot_h_bond_length_distribution(h_bond_df)


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
                elements_bond = [atom_a.name[0], atom_b.name[0]] # Get the first letter of the atom names, hopefully the element symbol
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
    bond_distances_df_shortened = bond_distances_df.iloc[-N_LAST_TIMESTEPS:]
    sns.lineplot(data=bond_distances_df_shortened, palette="tab10")
    plt.xlabel("Time Step")
    plt.ylabel("Bond Distance [Å]")
    # plt.title("Bond Distances Over Time")
    plt.legend(title="Bonds", bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig(title, dpi=DPI)

def plot_h_bond_length_distribution(h_bond_distances_df: pd.DataFrame, title: str = "h_bond_lengths_distribution.png"):
    # Plot the distribution of hydrogen bond lengths
    plt.figure(figsize=(10,10))
    sns.set_context(context="talk", font_scale=1.3)
    sns.histplot(h_bond_distances_df, x="Hydrogen Bond Length", hue="Bond Type", palette="tab10", multiple="stack", stat="probability", common_norm=True)
    plt.xlabel("Hydrogen Bond Length [Å]")
    plt.ylabel("Frequency")
    plt.title("Hydrogen Bond Length Distribution")
    plt.tight_layout()
    plt.savefig(title, dpi=DPI)

def plot_bond_length_distribution(h_bond_distances_df: pd.DataFrame, title: str = "bond_lengths_distribution.png"):
    # Plot the distribution of hydrogen bond lengths
    plt.figure(figsize=(10,10))
    sns.set_context(context="talk", font_scale=1.3)
    sns.histplot(h_bond_distances_df, x="Bond Length", hue="Bond Type", palette="tab10", multiple="stack", stat="probability", common_norm=True)
    plt.xlabel("Bond Length [Å]")
    plt.ylabel("Frequency")
    plt.title("Bond Length Distribution")
    plt.tight_layout()
    plt.savefig(title, dpi=DPI)

if __name__ == "__main__":
    main()