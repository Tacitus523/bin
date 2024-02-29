import os
import shutil

import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import distances
import numpy as np
import pandas as pd
import seaborn as sns

# Global parameters
DATA_DIR = "/data/user5/Sebastian/manual_umbrella/"
INITIAL_STRUCTURE = "geom_üü_sol.gro"


DISTANCE_IDENTIFIER_PAIRS = [("id 2", "id 7"), ("id 2", "id 12")]

NUM_BINS = 25  # Number of bins

# Constants
kj_mol_to_Hartree = 2625.5

# Input verfication
assert len(DISTANCE_IDENTIFIER_PAIRS) == 2, "We used the difference between to distance as CV here, so we need exactly two pairs"

def get_umbrella_trajectories(data_dir: str = os.getcwd()):
    """Gets trajectories and energies from Umbrellas

    Args:
        base_dir (str, optional): Folder with Walkers. Defaults to os.getcwd().
        
    Returns:
        List of umbrella_dirs, .tprs, .xtcs and .edrs 
    """
    
    umbrella_dirs = [os.path.join(data_dir, directory) for directory in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, directory)) and "UMBRELLA_" in directory]
    assert len(umbrella_dirs) > 0, "List of Umbrellas is empty"

    xtc_list = [os.path.join(umbrella_dir, file) for umbrella_dir in umbrella_dirs for file in os.listdir(umbrella_dir) if file.endswith(".xtc")]
    edr_list = [os.path.join(umbrella_dir, file) for umbrella_dir in umbrella_dirs for file in os.listdir(umbrella_dir) if file.endswith(".edr")]
    tpr_list = [os.path.join(umbrella_dir, file) for umbrella_dir in umbrella_dirs for file in os.listdir(umbrella_dir) if file.endswith(".tpr")]
    assert len(umbrella_dirs) == len(xtc_list), "Amount of Walker and .xtcs is different"
    assert len(umbrella_dirs) == len(edr_list), "Amount of Walker and .edrs is different"
    assert len(umbrella_dirs) == len(tpr_list), "Amount of Walker and .tprs is different"

    return umbrella_dirs, tpr_list, xtc_list, edr_list

def get_collective_variables(universe: mda.Universe, distance_pairs: list):
    """Reads collective variables from trajectories

    Args:
        universe (MDAnalysis.core.universe.Universe): trajectory as Universe 
        distance_pairs (list): list of 2er tuples of AtomGroups identifiers for each distance collective variable.
        
    Returns:
        np.ndarray: distance collective variables, shape = (n_CVs, n_frames)
    """
    assert len(distance_pairs) > 0, "You need at least one CV"
    n_CVs = len(distance_pairs) 
    
    # Prepare atom selection
    distance_pairs = [(universe.select_atoms(distance_pair[0]), universe.select_atoms(distance_pair[1])) for distance_pair in distance_pairs]
    
    # Initialize result lists
    distance_CVs = []
    for distance_pair in distance_pairs:
        distance_CVs.append([])
    
    # Iterate through trajectory, save CVs
    for time_step in universe.trajectory:
        #print(universe.trajectory.frame, universe.trajectory.time)
        for distance_index, distance_pair in enumerate(distance_pairs):
            resids1, resids2, distance = distances.dist(distance_pair[0], distance_pair[1], offset=0)
            distance_CVs[distance_index].append(distance)
    
    CVs = []
    for distance_CV in distance_CVs:
        CVs.append(np.concatenate(distance_CV, axis=-1))
    return CVs[1] - CVs[0]

umbrella_dirs, tpr_list, xtc_list, edr_list = get_umbrella_trajectories(DATA_DIR)

CVs_umbrellas = []
labels_umbrellas = []
for index, trajectory in enumerate(xtc_list):
    umbrella_dir = umbrella_dirs[index]
    initial_structure = INITIAL_STRUCTURE.replace("üü", os.path.basename(umbrella_dir).split("_")[1])
    initial_structure_path = os.path.join(umbrella_dirs[index], initial_structure)
    universe = mda.Universe(initial_structure_path, trajectory, refresh_offsets=True)
    print(f"Number of time steps in {os.path.basename(trajectory)}:", len(universe.trajectory))
    CVs_umbrella = get_collective_variables(universe, distance_pairs=DISTANCE_IDENTIFIER_PAIRS)
    labels_umbrella = [f"Umbrella {index}" for _ in range(CVs_umbrella.shape[0])]
    CVs_umbrellas.append(CVs_umbrella)
    labels_umbrellas += labels_umbrella
    
CVs_umbrellas = np.concatenate(CVs_umbrellas, axis=0)

df = pd.DataFrame({"CV": CVs_umbrellas, "labels": labels_umbrellas})

# Compute 1D histogram
sns.histplot(data=df, x="CV", hue="labels")
plt.title("1D Histogram of CV")
plt.legend([],[], frameon=False)
plt.savefig("Umbrella_CV_hist.png")
plt.close()