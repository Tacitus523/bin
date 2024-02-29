import os
import shutil

import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import distances
import numpy as np

# Global parameters
#DATA_DIR = "/data/user5/Sebastian/metadynamics/master/3_methyl/0.1V"
#DATA_DIR = "/data/user5/Sebastian/metadynamics/master/lukastest/test"
#DATA_DIR = "/data/user5/Sebastian/metadynamics/master/1_single_0V_water"
DATA_DIR = "/data/user5/Sebastian/Ergebnisse/structures/5_methyl_vac_0.1-1.0V/1.0V"
INITIAL_STRUCTURE = "geom_box.gro"
#INITIAL_STRUCTURE = "metad_start_c.gro"

TRAJECTORY_FILE = "traj_cat.xtc"
ENERGY_FILE = "energy_cat.xvg"

DISTANCE_IDENTIFIER_PAIRS = [("id 2", "id 7"), ("id 2", "id 12")]
#DISTANCE_IDENTIFIER_PAIRS = [("id 14", "id 36"), ("id 14", "id 58")]

NUM_BINS = 100  # Number of bins for each distance

ENERGY_MIN_DISPLAY = None # Minimum energy to display
ENERGY_MAX_DISPLAY = None # Maximal energy to display
# dist_max = relevant_CVs.max()
# dist_min = relevant_CVs.min()
dist_max = 5
dist_min = 1.7

# Constants
kj_mol_to_Hartree = 2625.5

def concat_walkers(data_dir: str = os.getcwd(), traj_target_file: str = "traj_cat.xtc", energy_target_file: str = "energy_cat.xvg"):
    """Concatenates trajectories and energies from Walkers into one trajectory and energy file

    Args:
        base_dir (str, optional): Folder with Walkers. Defaults to os.getcwd().
        traj_target_file (str, optional): Name of the file to write the concatenated trajectories to. Defaults to "traj_cat.xtc".
        energy_target_file (str, optional): Name of the file to write the concatenated energies to. Defaults to "energy_cat.xvg".
    """
    if os.path.isfile(traj_target_file) and os.path.isfile(energy_target_file):
        print(f"Already found {traj_target_file} and {energy_target_file}. Skipping concatenation")
        return
   
    gmx_d_path = shutil.which("gmx_d")
    assert gmx_d_path is not None, "Can't access gromacs commands"
    
    walker_list = [os.path.join(data_dir, directory) for directory in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, directory)) and "WALKER_" in directory]
    assert len(walker_list) > 0, "List of Walkers is empty"

    xtc_list = [os.path.join(walker, file) for walker in walker_list for file in os.listdir(walker) if file.endswith(".xtc")]
    edr_list = [os.path.join(walker, file) for walker in walker_list for file in os.listdir(walker) if file.endswith(".edr")]
    tpr_list = [os.path.join(walker, file) for walker in walker_list for file in os.listdir(walker) if file.endswith(".tpr")]
    assert len(walker_list) == len(xtc_list), "Amount of Walker and .xtcs is different"
    assert len(walker_list) == len(edr_list), "Amount of Walker and .edrs is different"
    assert len(walker_list) == len(tpr_list), "Amount of Walker and .tprs is different"
    
    # Make concatenated trajectories
    if not os.path.isfile(traj_target_file):
        os.system(f"gmx_d trjcat -f {' '.join(xtc_list)} -o {traj_target_file} -cat 1> /dev/null 2> /dev/null")
    else:
        print(f"Already found {traj_target_file}. Skipping concatenation")
    
    # Make concatenated energies
    if not os.path.isfile(energy_target_file):
        for index in range(len(edr_list)):
            os.system(f"echo -e '9\n\n' | gmx_d energy -f {edr_list[index]} -s {tpr_list[index]} -o temp_energy_{index}.xvg 1> /dev/null 2> /dev/null")
        
        energies = np.concatenate([np.loadtxt(f"temp_energy_{index}.xvg", skiprows=24, usecols=(1,), dtype=np.float32) for index in range(len(edr_list))], axis=0)/kj_mol_to_Hartree
        np.savetxt(energy_target_file, energies)
        os.system("rm temp_energy_*.xvg")
    else:
        print(f"Already found {energy_target_file}. Skipping concatenation")

def get_collective_variables(universe: mda.Universe, distance_pairs: list = None, custom_groups: tuple = None):
    """Reads collective variables from trajectories

    Args:
        universe (MDAnalysis.core.universe.Universe): trajectory as Universe 
        distance_pairs (list, optional): list of 2er tuples of AtomGroups identifiers for each distance collective variable. Defaults to None.
        custom_groups (tuple, optional): tuples of a function and a list of n-er tuples of AtomGroups
            as input for the function at each time step for each custom group. Defaults to None.
        
    Returns:
        list: distance and custom collective variables
    """
    
    # Replace dummy None
    if distance_pairs is None:
        distance_pairs = []
    if custom_groups is None:
        custom_groups = []
        
    assert len(distance_pairs) + len(custom_groups) > 0, "You need at least one CV"
    
    # Prepare atom selection
    distance_pairs = [(universe.select_atoms(distance_pair[0]), universe.select_atoms(distance_pair[1])) for distance_pair in distance_pairs]
    
    # Initialize result lists
    distance_CVs = []
    for distance_pair in distance_pairs:
        distance_CVs.append([])
    custom_CVs = []
    for custom_group in custom_groups:
        custom_CVs.append([])
    
    # Iterate through trajectory, save CVs
    for time_step in universe.trajectory:
        #print(universe.trajectory.frame, universe.trajectory.time)
        for distance_index, distance_pair in enumerate(distance_pairs):
            resids1, resids2, distance = distances.dist(distance_pair[0], distance_pair[1], offset=0)
            distance_CVs[distance_index].append(distance)
        
        for custom_index, custom_group in enumerate(custom_groups):
            custom_function = custom_group[0]
            custom_inputs = custom_group[1]
            result = custom_function(custom_inputs)
            custom_CVs[custom_index].append(result)
    
    CVs = []
    for distance_CV in distance_CVs:
        CVs.append(np.concatenate(distance_CV, axis=-1))
    for custom_CV in custom_CVs:
        CVs.append(np.concatenate(custom_CV, axis=-1))
    return np.array(CVs)

concat_walkers(DATA_DIR, TRAJECTORY_FILE, ENERGY_FILE)
initial_structure_path = os.path.join(DATA_DIR, INITIAL_STRUCTURE)
universe = mda.Universe(initial_structure_path, TRAJECTORY_FILE, refresh_offsets=True)
energies = np.loadtxt(ENERGY_FILE) # in Hartree

# S1 = universe.select_atoms("id 2")
# S2 = universe.select_atoms("id 7")
# S3 = universe.select_atoms("id 12")
# sulfurs = universe.select_atoms("name SG")
# print(S1,S2,S3)
# print(sulfurs)

print("Number of time steps:", len(universe.trajectory))

CVs = get_collective_variables(universe, distance_pairs=DISTANCE_IDENTIFIER_PAIRS)

# Currently just 2 CVs supported
relevant_CV_rows = (0,1)
assert len(relevant_CV_rows) == 2, "Currently just 2 CVs supported"

relevant_CVs = CVs[relevant_CV_rows, :]

# Compute 2D histogram
hist, x_edges, y_edges = np.histogram2d(relevant_CVs[0], relevant_CVs[1], bins=NUM_BINS, range=[[dist_min, dist_max], [dist_min, dist_max]])

masked_hist = np.ma.masked_array(hist, mask=hist == 0)

# Plot the 2D energy histogram
plt.imshow(masked_hist.T, origin='lower', extent=[dist_min, dist_max, dist_min, dist_max], cmap='coolwarm')
plt.colorbar(label='CV histogram')
plt.xlabel('Distance 1')
plt.ylabel('Distance 2')
plt.title('2D Histogram of CVs')
plt.savefig("CV_hist.png")
plt.close()

# Find the minimum energy in each bin
min_energy = np.zeros((NUM_BINS, NUM_BINS))
for i in range(NUM_BINS):
    for j in range(NUM_BINS):
        bin_entries = np.where((relevant_CVs[0] >= x_edges[i]) & (relevant_CVs[0] < x_edges[i + 1]) & 
                               (relevant_CVs[1] >= y_edges[j]) & (relevant_CVs[1] < y_edges[j + 1]))
        if len(bin_entries[0]) > 0:
            min_energy[i, j] = np.min(energies[bin_entries])

# Create masked array
masked_energy = np.ma.masked_array(min_energy, mask=min_energy == 0)

# Plot the 2D energy histogram
plt.imshow(masked_energy.T, origin='lower', extent=[dist_min, dist_max, dist_min, dist_max], vmin=ENERGY_MIN_DISPLAY, vmax=ENERGY_MAX_DISPLAY, cmap='viridis')
plt.colorbar(label='Minimum Energy')
plt.xlabel('Distance 1')
plt.ylabel('Distance 2')
plt.title('Minimum Energy in each bin')
plt.savefig("energy_hist.png")