#!/usr/bin/env python
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

#sys.path.append("/home/ka/ka_ipc/ka_he8978/kgcnn_fork")
sys.path.append("/home/lpetersen/kgcnn_fork")
from kgcnn.data.base import MemoryGraphDataset  # type: ignore
from kgcnn.utils import constants  # type: ignore

# DEFAULT VALUES
# DATA READ
# DATA_DIRECTORY: str = "/lustre/work/ws/ws1/ka_he8978-thiol_disulfide/training_data/B3LYP_aug-cc-pVTZ_vacuum" # Folder containing DATASET_NAME.kgcnn.pickle
# DATA_DIRECTORY: str = "/lustre/work/ws/ws1/ka_he8978-thiol_disulfide/training_data/B3LYP_aug-cc-pVTZ_water" # Folder containing DATASET_NAME.kgcnn.pickle
# DATASET_NAME: str = "ThiolDisulfidExchange" # Used in naming plots and looking for data
# DATA_DIRECTORY: str = "/lustre/work/ws/ws1/ka_he8978-dipeptide/training_data/B3LYP_aug-cc-pVTZ_water" # Folder containing DATASET_NAME.kgcnn.pickle
# DATA_DIRECTORY: str = "/lustre/work/ws/ws1/ka_he8978-dipeptide/training_data/B3LYP_aug-cc-pVTZ_vacuum" # Folder containing DATASET_NAME.kgcnn.pickle
DATA_DIRECTORY: str = "/data/lpetersen/training_data/alanindipeptid/B3LYP_aug-cc-pVTZ_vacuum"
DATASET_NAME: str = "Alanindipeptide" # Used in naming plots and looking for data

#TRAJECTORY_FILE: str = "run.xtc" # Path to the trajectory file
#TRAJECTORY_FILE: str = "traj_comp.xtc" # Path to the trajectory file
TRAJECTORY_FILE: str = "traj.xtc" # Path to the trajectory file
#TRAJECTORY_FILE: str = "dipeptid.xtc" # Path to the trajectory file
# TOPOLOGY_FILE: str = "geom.gro" # Path to the topology file (e.g., .gro or .pdb)
TOPOLOGY_FILE: str = "geom_box.gro" # Path to the topology file (e.g., .gro or .pdb)
#TOPOLOGY_FILE: str = "dipeptid.gro" # Path to the topology file (e.g., .gro or .pdb)

N_PLOTS: int = 5 # Number of plots to generate, chosen from the bonds with the largest, smallest values and standard deviation
N_LAST_TIMESTEPS: int = 0 # Number of last timesteps to plot, 0 for all
DPI: int = 100 # DPI for saving plots

# Backwards compatibility for missing TOPOLOGY_FILE, find and copy the starting structure
starting_idxs_file = "starting_structure_idxs.txt" # File containing the indices of the starting structure
starting_structures_dir = "start_geometries" # Directory containing the starting structures, expected to be inside DATA_DIRECTORY, only used when TOPOLOGY_FILE not present 
starting_structure_gro = "geom.gro" # Name of the starting structure file inside starting_structures_dir, only used when TOPOLOGY_FILE not present 
#starting_structure_gro = "sp_big_box.gro" # Name of the starting structure file inside starting_structures_dir, only used when TOPOLOGY_FILE not present 

default_collection_folder_name: str = "bond_distance_analysis"

def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze bond distances over time")
    ap.add_argument("-p", "--prefix", default=None, type=str, dest="prefix", action="store", required=False, help="Prefix of directionaries with trajectories, default: None", metavar="prefix")
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

    valid_dirs = []
    for dir in present_dirs:
        if not os.path.exists(os.path.join(dir, TRAJECTORY_FILE)):
            continue
        # Backwards compatibility for missing TOPOLOGY_FILE, find and copy the starting structure
        if not os.path.exists(os.path.join(dir, TOPOLOGY_FILE)):
            if not os.path.exists(os.path.join(dir, starting_idxs_file)):
                continue
            starting_idxs = np.loadtxt(os.path.join(dir,starting_idxs_file), dtype=int)
            if starting_idxs.ndim == 0:
                final_starting_idx = int(starting_idxs)
            else:
                final_starting_idx = int(starting_idxs[-1])
            starting_structure_dirs = next(os.walk(os.path.join(DATA_DIRECTORY, starting_structures_dir)))[1]
            gro_to_copy = [os.path.join(DATA_DIRECTORY, starting_structures_dir, dirname, starting_structure_gro) for dirname in starting_structure_dirs if f"{final_starting_idx}" in dirname][0]
            shutil.copy(gro_to_copy, os.path.join(dir, TOPOLOGY_FILE))
        valid_dirs.append(dir)
    valid_dirs = sorted(valid_dirs)
    print(f"Valid directories: {valid_dirs}")
    assert len(valid_dirs) > 0, "No valid directories found"

    data_directory = os.path.normpath(DATA_DIRECTORY)
    dataset = MemoryGraphDataset(data_directory=data_directory, dataset_name=DATASET_NAME)
    dataset.load()

    root_dir = os.getcwd()
    bond_distances_all_walkers = []
    for dir in valid_dirs:
        print(f"Analyzing {dir}")
        os.chdir(dir)
        bond_distances, edge_indices, atomic_numbers_bonds, elements_bonds = analyze_local_bond_distances(dataset, collection_folder_name=collection_folder_name)
        os.chdir(root_dir)
        bond_distances_all_walkers.append(bond_distances)
    
    bond_distances_all_walkers = np.concatenate(bond_distances_all_walkers, axis=0) # shape: (n_timesteps_all_walkers, n_atoms, n_atoms)
    # edge_indices, atomic_numbers_bonds, elements_bonds are expected to be the same for all walkers

    if collection_folder_name is not None:
        os.chdir(collection_folder_name)

    analyze_global_bond_distances(bond_distances_all_walkers, edge_indices, atomic_numbers_bonds, elements_bonds) 


def analyze_local_bond_distances(dataset: MemoryGraphDataset, collection_folder_name: str = None) -> None:
    try:
        starting_idxs = np.loadtxt(starting_idxs_file, dtype=int)
        if starting_idxs.ndim == 0:
            final_starting_idx = int(starting_idxs)
        else:
            final_starting_idx = int(starting_idxs[-1])
    except FileNotFoundError:
        print(f"{os.getcwd()}: No starting_idxs file found for bond connectivity, defaulting to 0")
        final_starting_idx = 0

    n_atoms, atomic_numbers, unique_edge_indices, atomic_numbers_bonds, elements_bonds = get_atomic_numbers_and_elements(dataset, final_starting_idx)

    try:
        u = mda.Universe(TOPOLOGY_FILE, TRAJECTORY_FILE)
    except Exception as e:
        print(f"{os.getcwd()}: Problem with Topology, skipping analysis", file=sys.stderr)
        print(e, file=sys.stderr)
        return

    qm_atoms = u.atoms[:n_atoms]

    distance_matrices = []
    for timestep in u.trajectory:
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
            bond_distances_df["Time Step"] = np.arange(len(u.trajectory))
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


def get_atomic_numbers_and_elements(dataset: MemoryGraphDataset, final_starting_idx: int) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the atomic numbers and elements of the atoms in the bonds.

    Args:
        dataset (MemoryGraphDataset): The dataset containing the graph data.
        final_starting_idx (int): The index of the final starting point.

    Returns:
        Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing the following:
            - n_atoms (int): The number of atoms.
            - atomic_numbers (np.ndarray): An array of atomic numbers.
            - unique_edge_indices (np.ndarray): An array of unique edge indices.
            - atomic_numbers_bonds (np.ndarray): An array of atomic numbers of the atoms in the bonds.
            - elements_bonds (np.ndarray): An array of elements of the atoms in the bonds.
    """
    # Get the atomic numbers and elements of the atoms in the bonds
    atomic_number_element_dict = constants.atomic_number_to_element
    atomic_numbers = np.array(dataset[final_starting_idx].get("node_number")).flatten()
    n_atoms = len(atomic_numbers)

    target_edge_indices = dataset[final_starting_idx]["edge_indices"]
    unique_edge_indices = []
    for edge in target_edge_indices:
        if [edge[0], edge[1]] not in unique_edge_indices and [edge[1], edge[0]] not in unique_edge_indices:
            unique_edge_indices.append([edge[0], edge[1]])
    unique_edge_indices = np.array(unique_edge_indices) # Get the unique edge indices, shape: (n_bonds, 2)

    atomic_numbers_bonds = atomic_numbers[unique_edge_indices] # Get the respective atomic numbers of the atoms in the bonds, shape: (n_bonds, 2)
    elements_bonds = np.array(
        [[atomic_number_element_dict[atomic_number] for atomic_number in atomic_numbers_bond] for atomic_numbers_bond in atomic_numbers_bonds]
    ) # Get the respective elements of the atoms in the bonds, shape: (n_bonds, 2)

    return n_atoms, atomic_numbers, unique_edge_indices, atomic_numbers_bonds, elements_bonds

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