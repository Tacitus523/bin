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
# DATA_DIRECTORY = "/lustre/work/ws/ws1/ka_he8978-thiol_disulfide/training_data/B3LYP_aug-cc-pVTZ_vacuum" # Folder containing DATASET_NAME.kgcnn.pickle
# DATA_DIRECTORY = "/lustre/work/ws/ws1/ka_he8978-thiol_disulfide/training_data/B3LYP_aug-cc-pVTZ_water" # Folder containing DATASET_NAME.kgcnn.pickle
# DATASET_NAME = "ThiolDisulfidExchange" # Used in naming plots and looking for data
DATA_DIRECTORY = "/lustre/work/ws/ws1/ka_he8978-dipeptide/training_data/B3LYP_aug-cc-pVTZ_water" # Folder containing DATASET_NAME.kgcnn.pickle
#DATA_DIRECTORY = "/lustre/work/ws/ws1/ka_he8978-dipeptide/training_data/B3LYP_aug-cc-pVTZ_water" # Folder containing DATASET_NAME.kgcnn.pickle
DATASET_NAME = "Alanindipeptide" # Used in naming plots and looking for data

TRAJECTORY_FILE = "run.xtc" # Path to the trajectory file
TOPOLOGY_FILE = "run.gro" # Path to the topology file (e.g., .gro or .pdb)

N_PLOTS = 5 # Number of plots to generate, chosen from the bonds with the largest standard deviation
DPI = 100 # DPI for saving plots

starting_idxs_file = "starting_structure_idxs.txt" # File containing the indices of the starting structure
starting_structures_dir = "start_geometries" # Directory containing the starting structures, expected to be inside DATA_DIRECTORY, only used when TOPOLOGY_FILE not present 
starting_structure_gro = "sp_big_box.gro" # Name of the starting structure file inside starting_structures_dir, only used when TOPOLOGY_FILE not present 

def main():
    ap = argparse.ArgumentParser(description="Analyze bond distances over time")
    ap.add_argument("-d", "--dir", default=None, type=str, dest="dir", action="store", required=False, help="Directionary with trajectories, default: None", metavar="directionary")
    args = ap.parse_args()
    if args.dir is None:
        present_dirs = [os.getcwd()]
    else:
        os.chdir(args.dir)
        present_dirs = [dir for dir in os.listdir(".") if os.path.isdir(dir)]
    
    valid_dirs = []
    for dir in present_dirs:
        if not os.path.exists(os.path.join(dir, starting_idxs_file)):
            continue
        if not os.path.exists(os.path.join(dir, TRAJECTORY_FILE)):
            continue
        if not os.path.exists(os.path.join(dir, TOPOLOGY_FILE)):
            # Backwards compatibility for missing TOPOLOGY_FILE, find and copy the starting structure
            starting_idxs = np.loadtxt(os.path.join(dir,starting_idxs_file), dtype=int)
            final_starting_idx = int(starting_idxs[-1])
            starting_structure_dirs = next(os.walk(os.path.join(DATA_DIRECTORY, starting_structures_dir)))[1]
            gro_to_copy = [os.path.join(DATA_DIRECTORY, starting_structures_dir, dirname, starting_structure_gro) for dirname in starting_structure_dirs if f"{final_starting_idx}" in dirname][0]
            shutil.copy(gro_to_copy, os.path.join(dir, TOPOLOGY_FILE))
        valid_dirs.append(dir)

    data_directory = os.path.normpath(DATA_DIRECTORY)
    dataset = MemoryGraphDataset(data_directory=data_directory, dataset_name=DATASET_NAME)
    dataset.load()

    root_dir = os.getcwd()
    for dir in valid_dirs:
        os.chdir(dir)
        analyze_bond_distances(dataset)
        os.chdir(root_dir)

def analyze_bond_distances(dataset):
    starting_idxs = np.loadtxt(starting_idxs_file, dtype=int)
    final_starting_idx = int(starting_idxs[-1])

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
    
    all_atoms = u.select_atoms("all")

    bond_distances_all_timesteps = []
    for timestep in u.trajectory:
        # Calculate the distance matrix
        distance_matrix = mda.lib.distances.distance_array(all_atoms.positions, all_atoms.positions)
        
        # Extract bond_distances from the distance matrix using the specified edge indices
        bond_distances = distance_matrix[filtered_edge_indices[:, 0], filtered_edge_indices[:, 1]]
        bond_distances_all_timesteps.append(bond_distances)

    bond_distances_all_timesteps = np.array(bond_distances_all_timesteps)
    bond_distance_stds = bond_distances_all_timesteps.std(axis=0)
    largest_bond_distance_std_indices = np.argsort(bond_distance_stds, )[-N_PLOTS:][::-1] # Only plot the N_PLOTS bonds with the largest standard deviation
    largest_std_edge_indices = filtered_edge_indices[largest_bond_distance_std_indices]
    largest_std_bond_distances = bond_distances_all_timesteps[:, largest_bond_distance_std_indices]
    del filtered_edge_indices
    del bond_distances_all_timesteps

    atomic_number_element_dict = constants.atomic_number_to_element
    atomic_numbers = np.array(dataset[final_starting_idx].get("node_number")).flatten()
    atomic_numbers = atomic_numbers[largest_std_edge_indices.flatten()]
    elements = [atomic_number_element_dict[num] for num in atomic_numbers]
    elements = np.array(elements).reshape(-1, 2)
    labels = [f"({elements[i, 0]}{largest_std_edge_indices[i, 0]}, {elements[i, 1]}{largest_std_edge_indices[i, 1]})" for i in range(len(largest_std_edge_indices))]
    bond_distances_df = pd.DataFrame(largest_std_bond_distances, columns=labels)
    bond_distances_df["Time Step"] = np.arange(len(u.trajectory))
    bond_distances_df = bond_distances_df.set_index("Time Step")

    plot_bond_distances(bond_distances_df)

def plot_bond_distances(bond_distances_df):
    # Plot the bond distances over time
    plt.figure(figsize=(10,10))
    sns.set_context(context="talk", font_scale=1.3)
    sns.lineplot(data=bond_distances_df, palette="tab10")
    plt.xlabel("Time Step")
    plt.ylabel("Bond Distance [Å]")
    # plt.title("Bond Distances Over Time")
    plt.legend(title="Bonds", bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig("bond_distances_vs_time_steps.png", dpi=DPI)

if __name__ == "__main__":
    main()