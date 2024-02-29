import os
import shutil

import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import distances
import numpy as np

DATA_DIR = "/data/lpetersen/thiol_disulfide/B3LYP_aug-cc-pVTZ_water/scan"
INITIAL_STRUCTURE = "GEOM_00000/geom_box.gro"
TRAJECTORY_FILE = "geoms.xyz"
ENERGY_FILE = "energies.txt"
CHARGE_FILE = "charges_hirsh.txt"

trajectory_file   = os.path.join(DATA_DIR, TRAJECTORY_FILE)
initial_structure = os.path.join(DATA_DIR, INITIAL_STRUCTURE)
energy_file_water = os.path.join(DATA_DIR, ENERGY_FILE)
charge_file_water = os.path.join(DATA_DIR, CHARGE_FILE)

def get_constrained_indices(traj_file: str, init_struct: str, identifier: str|list, boundaries: list):
    """Gets the indices of structures with distances within the boundary

    Args:
        traj_file (str): trajectory file with structures
        init_struct (str): example .gro with one structure
        identifier (str | list): MDA identifier for atoms to calculate the distances from
        boundaries (list): upper and lower boundary of allowed distances
    
    Returns:
        np.array(bool): Boolean array of shape (n_frames,), structures fulfilling the condition are true 
    """
    
    
    assert boundaries[0] < boundaries[1]
    
    universe = mda.Universe(init_struct, traj_file)
    sulfurs = universe.select_atoms(identifier)

    # Calculate distances for each frame
    distances = []
    for frame in universe.trajectory:
        frame_distances = mda.lib.distances.distance_array(sulfurs, sulfurs)
        upper_traingle_indices = np.triu_indices(frame_distances.shape[0], k=1)
        upper_triangle_flattened = frame_distances[upper_traingle_indices].reshape(1,-1)
        distances.append(upper_triangle_flattened)

    # Concatenate distances from all frames
    all_distances = np.concatenate(distances, axis=0)

    # Find frames where any two distances are within the threshold
    lower_boundary = boundaries[0]
    upper_boundary = boundaries[1]
    frames_within_boundary = np.sum(np.logical_and(all_distances >= lower_boundary, all_distances <= upper_boundary), axis=1) >= 2
    
    return frames_within_boundary


if __name__ == "__main__":
    identifier = "name SG"
    boundaries = [2.3, 2.7]
    constrained_indices = get_constrained_indices(trajectory_file, initial_structure, identifier, boundaries)
    print("len:", np.sum(constrained_indices))