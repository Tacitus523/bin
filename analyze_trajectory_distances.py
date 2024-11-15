#!/home/ka/ka_ipc/ka_he8978/miniconda3/envs/kgcnn_new/bin/python
import argparse
import numpy as np
import os
import pandas as pd
import shutil
import sys
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import MDAnalysis as mda
import seaborn as sns

sys.path.append("/home/ka/ka_ipc/ka_he8978/kgcnn_fork")
from kgcnn.data.base import MemoryGraphDataset  # type: ignore
from kgcnn.utils import constants  # type: ignore

# DEFAULT VALUES
# DATA READ
# DATA_DIRECTORY: str = "/lustre/work/ws/ws1/ka_he8978-thiol_disulfide/training_data/B3LYP_aug-cc-pVTZ_vacuum" # Folder containing DATASET_NAME.kgcnn.pickle
# DATA_DIRECTORY: str = "/lustre/work/ws/ws1/ka_he8978-thiol_disulfide/training_data/B3LYP_aug-cc-pVTZ_water" # Folder containing DATASET_NAME.kgcnn.pickle
# DATASET_NAME: str = "ThiolDisulfidExchange" # Used in naming plots and looking for data
# DATA_DIRECTORY: str = "/lustre/work/ws/ws1/ka_he8978-dipeptide/training_data/B3LYP_aug-cc-pVTZ_water" # Folder containing DATASET_NAME.kgcnn.pickle
DATA_DIRECTORY: str = "/lustre/work/ws/ws1/ka_he8978-dipeptide/training_data/B3LYP_aug-cc-pVTZ_vacuum" # Folder containing DATASET_NAME.kgcnn.pickle
DATASET_NAME: str = "Alanindipeptide" # Used in naming plots and looking for data

#TRAJECTORY_FILE: str = "run.xtc" # Path to the trajectory file
TRAJECTORY_FILE: str = "traj_comp.xtc" # Path to the trajectory file
TOPOLOGY_FILE: str = "geom.gro" # Path to the topology file (e.g., .gro or .pdb)

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

    data_directory = os.path.normpath(DATA_DIRECTORY)
    dataset = MemoryGraphDataset(data_directory=data_directory, dataset_name=DATASET_NAME)
    dataset.load()

    root_dir = os.getcwd()
    for dir in valid_dirs:
        print(f"Analyzing {dir}")
        os.chdir(dir)
        analyze_bond_distances(dataset, collection_folder_name=collection_folder_name)
        os.chdir(root_dir)


def analyze_bond_distances(dataset: MemoryGraphDataset, collection_folder_name: str = None) -> None:
    try:
        starting_idxs = np.loadtxt(starting_idxs_file, dtype=int)
        if starting_idxs.ndim == 0:
            final_starting_idx = int(starting_idxs)
        else:
            final_starting_idx = int(starting_idxs[-1])
    except FileNotFoundError:
        print(f"{os.getcwd()}: No starting_idxs file found for bond connectivity, defaulting to 0")
        final_starting_idx = 0

    # Get the atomic numbers and elements of the atoms in the bonds
    atomic_number_element_dict = constants.atomic_number_to_element
    atomic_numbers = np.array(dataset[final_starting_idx].get("node_number")).flatten()
    n_atoms = len(atomic_numbers)

    edge_indices = dataset[final_starting_idx]["edge_indices"]
    filtered_edge_indices = []
    for edge in edge_indices:
        if [edge[0], edge[1]] not in filtered_edge_indices and [edge[1], edge[0]] not in filtered_edge_indices:
            filtered_edge_indices.append([edge[0], edge[1]])
    filtered_edge_indices = np.array(filtered_edge_indices)

    try:
        u = mda.Universe(TOPOLOGY_FILE, TRAJECTORY_FILE)
    except Exception as e:
        print(f"{os.getcwd()}: Problem with Topology, skipping analysis", file=sys.stderr)
        print(e, file=sys.stderr)
        return

    qm_atoms = u.atoms[:n_atoms]

    bond_distances_all_timesteps = []
    for timestep in u.trajectory:
        # Calculate the distance matrix
        distance_matrix = mda.lib.distances.distance_array(qm_atoms.positions, qm_atoms.positions)
        
        # Extract bond_distances from the distance matrix using the specified edge indices
        bond_distances = distance_matrix[filtered_edge_indices[:, 0], filtered_edge_indices[:, 1]]
        bond_distances_all_timesteps.append(bond_distances)

    bond_distances_all_timesteps = np.array(bond_distances_all_timesteps)
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

            edge_indices = filtered_edge_indices[indices]
            atomic_numbers_indices = atomic_numbers[edge_indices.flatten()]
            elements = [atomic_number_element_dict[num] for num in atomic_numbers_indices]
            elements = np.array(elements).reshape(-1, 2)

            labels = [f"({elements[i, 0]}{edge_indices[i, 0]}, {elements[i, 1]}{edge_indices[i, 1]})" for i in range(len(edge_indices))]
            bond_distances_df = pd.DataFrame(extreme_bond_data, columns=labels)
            bond_distances_df["Time Step"] = np.arange(len(u.trajectory))
            bond_distances_df = bond_distances_df.set_index("Time Step")
            title = f"{data_label}_{size_label}_vs_time_steps.png"
            plot_bond_distances(bond_distances_df, title=title)
    
            if collection_folder_name is not None:
                folder_name = os.path.basename(os.getcwd())
                collection_title = f"{data_label}_{size_label}_vs_time_steps_{folder_name}.png"
                shutil.copy(title, os.path.join(collection_folder_name, collection_title))


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

if __name__ == "__main__":
    main()