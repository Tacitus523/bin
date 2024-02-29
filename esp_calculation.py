import os

import numpy as np
from scipy.spatial.distance import cdist

# Input
CHARGE_FILE = "/home/tkubar/GMX-DFTB/gromacs-dftbplus/release-machine-learning/share/gromacs/top/amber99sb.ff/tip3p.itp"  # Name of topology file in which mm atoms are defined (itp or top)
TPR_FILE = "sp.tpr" # Portable binary run input file containing both topology and coordinate information, .tpr
TRJ_FILE = "sp.trr" # Trajectory file, .trr/.xtc
# CHARGE_FILE = "/data/user6/menns/fr0/inwater/gs/tip3p.itp"  # Name of topology file in which mm atoms are defined (itp or top)
# TPR_FILE = "/data/user6/menns/fr0/inwater/gs/md_gstate1.tpr" # Portable binary run input file containing both topology and coordinate information, .tpr
# TRJ_FILE = "/data/user6/menns/fr0/inwater/gs/md_gstate1.xtc" # Trajectory file, .trr/.xtc
QM_GROUP = 1 # Input to gmx_d make_idx to access the QM-region
MM_GROUP = 13 # Input to gmx_d make_idx to access the MM-region

#Constants
echarge = 1.602176634e-19
eps0 = 8.8541878128e-12
atomic_units_to_volt = 27.211386245988
angstrom_to_bohr = 1.8897259886

charge_file = CHARGE_FILE
tpr_file = TPR_FILE
trj_file = TRJ_FILE

# Begin readout of forcefield charges
with open(charge_file, "r") as f:
    charge_file_content = f.read()

atom_charges = {}
i=1 # skip line with column headers
while True:
    atom_line = charge_file_content.split("[ atoms ]\n")[1].split("\n")[i] # line with atom properties
    if not atom_line: # break on empty string
        break
    atom_line = atom_line.split()
    atom_name = atom_line[4]
    atom_charge = float(atom_line[6])
    atom_charges[atom_name] = atom_charge
    i+=1
# End readout of forcefield charges

# Generate multistep QM .gro from trajectory
qm_index_command = f"""gmx_d make_ndx -f {tpr_file} -o temp_qm_index << EOF > /dev/null 2>&1
keep {QM_GROUP}
q
EOF""" # generates an index-file for the QM-region and redirects the gromacs output to nowhere
os.system(qm_index_command) # executes the command
os.system(f"gmx_d trjconv -f {trj_file} -s {tpr_file} -n temp_qm_index -o temp_qm_region.gro > /dev/null 2>&1") # get the qm-region .gro from the trajectory

# Generate mutistep MM .gro from trajectory
mm_index_command = f"""gmx_d make_ndx -f {tpr_file} -o temp_mm_index << EOF > /dev/null 2>&1
keep {MM_GROUP}
q
EOF""" # generates an index-file for the MM-region and redirects the gromacs output to nowhere
os.system(mm_index_command) # executes the command
os.system(f"gmx_d trjconv -f {trj_file} -s {tpr_file} -n temp_mm_index -o temp_mm_region.gro > /dev/null 2>&1") # get the mm-region .gro from the trajectory

# get line-numbers, three additional lines per step(comment, atom number and box vector), therefore atoms + 3
with open("temp_qm_region.gro", "r") as qm_file, open("temp_mm_region.gro", "r") as mm_file:
    qm_file.readline(), mm_file.readline()
    n_qm_atoms, n_mm_atoms = int(qm_file.readline()), int(mm_file.readline())
    n_qm_lines, n_mm_lines = n_qm_atoms+3, n_mm_atoms+3

# go trough all frames, make temporary step.gro, and calculate esp for each step
with open("temp_qm_region.gro", "r") as qm_file, open("temp_mm_region.gro", "r") as mm_file:
    qm_lines = [qm_file.readline() for _ in range(n_qm_lines)]
    mm_lines = [mm_file.readline() for _ in range(n_mm_lines)]
    
    esps = []
    while qm_lines[0] and mm_lines[0]: # breaks if first entry is empty string
        with open("temp_qm_step.gro", "w") as qm_step_file, open("temp_mm_step.gro", "w") as mm_step_file:
            qm_step_file.writelines(qm_lines)
            mm_step_file.writelines(mm_lines)
            
        qm_coords = np.genfromtxt("temp_qm_step.gro", skip_header=2, skip_footer=1, usecols=(3,4,5))
        
        mm_coords = np.genfromtxt("temp_mm_step.gro", skip_header=2, skip_footer=1, usecols=(3,4,5))
        mm_types = np.genfromtxt("temp_mm_step.gro", skip_header=2, skip_footer=1, usecols=1, dtype=str)
        mm_charges = np.array([atom_charges[atom_type] for atom_type in mm_types]).reshape((-1,1))
        
        qmmm_distances = cdist(qm_coords, mm_coords) # shape: (n_qm_atoms, n_mm_atoms)
        esps_step = np.matmul(1/qmmm_distances, mm_charges).reshape((1,-1))
        esps.append(esps_step)
        
        qm_lines = [qm_file.readline() for _ in range(n_qm_lines)]
        mm_lines = [mm_file.readline() for _ in range(n_mm_lines)]


esps = np.concatenate(esps, axis=0)
esps = esps/angstrom_to_bohr/10 # from e/nm to e/bohr(atomic unit)
np.savetxt("esps_by_mm.txt", esps, fmt='%3.6f')

# Cleanup
os.remove("temp_qm_index.ndx")
os.remove("temp_mm_index.ndx")
os.remove("temp_qm_region.gro")
os.remove("temp_mm_region.gro")
os.remove("temp_qm_step.gro")
os.remove("temp_mm_step.gro")