import os
import re
import shutil
import subprocess

import pandas as pd
import MDAnalysis as mda
import numpy as np

N_GEOMS = 10000
DATA_FOLDER = "/data/lpetersen/thiol_disulfide/B3LYP_aug-cc-pVTZ_protein/trajectories"
FOLDERS = ["01_OxMo1", "02_ReDi1", "03_ReDi2X7d1", "04_ReMo2X7"] # Folders with trajectories, relative or absolute path
XTC_FILES = [
   "F_500K_std_1.xtc",
   "F_500K_std_2.xtc",
   "F_500K_std_3.xtc",
   "F_500K_std_4.xtc"
] # Trajectory files

TPR_FILES = [
   "F_500K_std_1.tpr",
   "F_500K_std_2.tpr",
   "F_500K_std_3.tpr",
   "F_500K_std_4.tpr"
] # Combined topology/parameter files
# MOL_COUNTS = [10093, 2433, 2506, 3108] # Geometries per trajectory
# GROUP_COUNTS = [8488, 44752, 39953, 28958] # Groups per geometry
# AT_COUNTS = [26879, 137142, 122744, 89798] # Atoms per geometry
MM_LINK_INDEX_LISTS = [[346,1682,1717], [346,2103,2113], [391,346,2102], [3233,346,2103]]
QM_LINK_INDEX_LISTS = [[348,1684,1719], [348,2105,2115], [393,348,2104], [3235,348,2105]]
SINGLEPOINT_FILES = { # Mostly supposed to be in the folder, that this file is executed from
    "topology": "topol_with_la.top", # Taken from FOLDERS, don't give the path, becomes a list
    "index": "index.ndx", # None, if not requiered, taken from FOLDERS
    "orca_basename": "sp", # Will define the name of the .out files
    "mdp": "sp.mdp", 
    "orcainfo": "sp.ORCAINFO" # Will be adjusted to to fit orca_basename
}

def run_sp_calc_from_classic_traj(N_GEOMS, DATA_FOLDER, FOLDERS, XTC_FILES, TPR_FILES, MM_LINK_INDEX_LISTS, QM_LINK_INDEX_LISTS, SINGLEPOINT_FILES: dict):
    check_assertions(N_GEOMS, FOLDERS, XTC_FILES, TPR_FILES, MM_LINK_INDEX_LISTS)
    
    # Prepare path in SINGLEPOINT_FILES
    for key, value in SINGLEPOINT_FILES.items():
        if key in ["mdp", "orcainfo"] and value is not None:
            SINGLEPOINT_FILES[key] = clean_path(os.path.abspath(value))
        if key in ["index", "topology"] and value is not None:
            SINGLEPOINT_FILES[key] = [clean_path(os.path.join(DATA_FOLDER, folder, SINGLEPOINT_FILES[key])) for folder in FOLDERS]
        if key==["index"] and value is None:
            SINGLEPOINT_FILES[key] = [None for _ in FOLDERS]
    
    # Prepare data structure
    sp_calc_path = "sp_calculations"
    create_folder(sp_calc_path, ask_overwrite=True)
    
    # Create universes
    universes = []
    zipper = zip(FOLDERS, XTC_FILES, TPR_FILES)
    for folder, xtc_file, tpr_file in zipper:
        xtc_file = os.path.join(DATA_FOLDER, folder, xtc_file)
        tpr_file = os.path.join(DATA_FOLDER, folder, tpr_file)
        universe = mda.Universe(tpr_file, xtc_file)
        universes.append(universe)
        
    # Random number generation in sets of n
    universe_lengths = [len(universe.trajectory) for universe in universes]
    total_length = sum(universe_lengths)
    assert total_length > N_GEOMS, f"The amount of demanded geometries({N_GEOMS}) is longer than the amount of supplied geometries({total_length})"
    fractions = [int(N_GEOMS*universe_length/total_length) for universe_length in universe_lengths]
    N_GEOMS = sum(fractions)
    chosen_timestep_list = [np.random.choice(universe_length, size=(fraction,), replace=False) for universe_length, fraction in zip(universe_lengths,fractions)]

    os.chdir(sp_calc_path)
    zipper = zip(universes, chosen_timestep_list, MM_LINK_INDEX_LISTS)
    geom_index = 0
    for universe_index, (universe, chosen_timesteps, mm_link_indices) in enumerate(zipper):
        mm_link_indices.sort() # following indices in the list will be incremented, so the list has to be ordered
        # # Conversion with gromacs
        # traj_conversion = "echo 0 | gmx_d trjconv -f *.xtc -s *.tpr -o temp.gro &> /dev/null"
        # subprocess.run(traj_conversion, shell=True)
        
        # Conversion with MDAnalysis
        atoms = universe.select_atoms("all")
        dimensions = universe.dimensions
        
        
        base_df = make_base_df(atoms)
        
        if universe_index > 0:
            base_df["resnum"] += 1
        
        # Filter Stuff from trajectory
        chosen_frames = universe.trajectory[chosen_timesteps]
        for frame_index, frame in enumerate(chosen_frames):
            n_atoms = atoms.n_atoms
            frame_df = make_frame_df(base_df, atoms)
            # LA insertation into classical MM trajectories
            for mm_link_index in mm_link_indices:
                frame_df = insert_linker_atom(frame_df, mm_link_index) # Increment the mm_link_index to consider previous insertations
                n_atoms += 1
                
            # Names for geometry and data structure generation
            geom_name = f"GEOM_{str(geom_index).rjust(len(str(N_GEOMS)),'0')}" # check maximum amount of geoms and buffer with 0
            geom_filename = f"geom.gro"
            geom_foldername = geom_name
            create_folder(geom_foldername)
            os.chdir(geom_foldername)
            
            # Finish the creation of the .gro with Linker atoms
            comment = f"{os.path.join(FOLDERS[universe_index], XTC_FILES[universe_index])}, time step {chosen_timesteps[frame_index]}\n" # Trajectory origin and its timestep
            write_gro(geom_filename, comment, n_atoms, frame_df, dimensions)
            
            # run singlepoint calculations
            topol_filename = SINGLEPOINT_FILES["topology"][universe_index]
            index_filename = SINGLEPOINT_FILES["index"][universe_index]
            run_single_point_calculation(geom_name, geom_filename, topol_filename, index_filename, SINGLEPOINT_FILES)

            link_system_specific_files(topol_filename, index_filename)
            geom_index += 1
            os.chdir("..")
            
    os.chdir("..")
    write_index_file(N_GEOMS, FOLDERS, XTC_FILES, chosen_timestep_list)
        

def split_group_name(group_name):
    """Gets the numbers and the letters from a group name.
        e.g. "8482SOL" --> (8482, "SOL")
    """
    pattern = r"(\d+)([a-zA-Z]+)"
    match = re.search(pattern, group_name)
    numbers = int(match.group(1))
    letters = match.group(2)
    return numbers, letters

def split_type_name(type_name):
    """Gets the numbers and the letters from a type name.
        e.g. "HW126856" --> (8482, "SOL")
    """
    pattern = r"([a-zA-Z]+)(\d+)"
    match = re.search(pattern, type_name)
    letters = match.group(1)
    numbers = int(match.group(2))
    return numbers, letters

def insert_linker_atom(frame_df, index):
    mm_entry = frame_df[frame_df["id"]==index] # This is still a Dataframe, not a Series
    assert len(mm_entry)==1, "Unique atom ids are expected" # Make sure the index is unique
    # Create data for the linker atom
    la_entry = mm_entry.copy()
    la_entry["name"] = "LA"
    la_entry["resname"] = "XXX"
    first_solvent_id = min(frame_df[frame_df["resname"]=="SOL"].index) # Identify the first solvent id
    first_solvent_entry = frame_df.iloc[first_solvent_id]
    la_entry["id"] = first_solvent_entry["id"]
    la_entry["resnum"] = first_solvent_entry["resnum"]
    la_entry[["x","y","z"]] = la_entry[["x","y","z"]] + 0.01
    # LA insertion and incrementation, increment the indices after the cut
    pre_cut_df = frame_df.iloc[:first_solvent_id].copy() # Cut the frame_df before the residue of the MM atom
    post_cut_df = frame_df.iloc[first_solvent_id:].copy() # Cut the frame_df after the residue of the MM atom
    post_cut_df[["resnum","id"]] += 1 # Increment residue index and atom index
    frame_df = pd.concat([pre_cut_df, la_entry, post_cut_df], axis=0, ignore_index=True) # Concat df cuts
    return frame_df
    
def make_base_df(atoms: mda.AtomGroup):
    df = pd.DataFrame()
    df["resnum"] = atoms.resnums
    df["resname"] = atoms.resnames
    df["name"] = atoms.names
    df["id"] = (atoms.ids+1)
    return df

def make_frame_df(base_df: pd.DataFrame, atoms: mda.AtomGroup):
    pos_df = pd.DataFrame(atoms.positions/10, columns=["x","y","z"]) # MDAnalysis units in A, gro Units in nm
    frame_df = pd.concat([base_df, pos_df], axis=1)
    return frame_df

def write_gro(filename: str, comment: str, n_atoms: int, frame_df: pd.DataFrame, dimensions: list):
    frame_df["resnum"] = frame_df["resnum"] % 100000 # Modulo operator to prevent number with more than 5 digits
    frame_df["id"] = frame_df["id"] % 100000 # Modulo operator to prevent number with more than 5 digits
    
    # Custom formatting functions for each column
    def format_int(val):
        return f"{val:>5d}"

    def format_str_left_align(val):
        return f"{val:<5s}"
    
    def format_str_right_align(val):
        return f"{val:>5s}"

    def format_float(val):
        return f"{val:>8.3f}"

    # Apply the formatters to specific columns
    formatters = {
        "resnum": format_int,
        "resname": format_str_left_align,
        "name": format_str_right_align,
        "id": format_int,
        "x": format_float,
        "y": format_float,
        "z": format_float
    }
    for col, formatter in formatters.items():
        frame_df[col] = frame_df[col].apply(formatter)
 
    # Format the gro contents
    gro_string = frame_df.to_csv(header=False, index=False, sep='ß') # Use ß as a hopefully unused placeholder separator
    gro_string = re.sub('ß', '', gro_string) # remove the separator
    
    # Format other content
    n_atoms_string = f"{n_atoms}\n"
    dimensions = [f"{dimension/10:4.5f}" for dimension in dimensions[:3]]# With conversion from A to nm
    dimensions_string = f"{dimensions[0]:>10s}{dimensions[1]:>10s}{dimensions[2]:>10s}\n" 
    
    with open(filename, "w") as f:
        f.write(comment)
        f.write(n_atoms_string)
        f.write(gro_string)
        f.write(dimensions_string)

def link_system_specific_files(topol_filename: str, index_filename: str):
    topol_linkname = "topol.top"
    os.symlink(topol_filename, topol_linkname)
    if index_filename is not None:
        index_linkname = "index.ndx"
        os.symlink(index_filename, index_linkname)
        
def write_index_file(N_GEOMS, FOLDERS, XTC_FILES, timestep_list):
    index_file_name = "geom_indices.txt"
    if os.path.exists(index_file_name):
        # Delete the file
        os.remove(index_file_name)

    f = open(index_file_name, "a")
    geom_index = 0
    zipper = zip(FOLDERS, XTC_FILES, timestep_list)
    for folder, xtc_file, timesteps in zipper:
        traj_origin = os.path.join(folder, xtc_file)
        for timestep in timesteps:
            geom_name = f"GEOM_{str(geom_index).rjust(len(str(N_GEOMS)),'0')}"
            f.write(f"{geom_name}: {traj_origin}, {timestep}\n")
            geom_index += 1
    f.close()
        
def check_assertions(N_GEOMS, FOLDERS, XTC_FILES, TPR_FILES, MM_LINK_INDEX_LISTS):
    assert all(len(lst) == len(FOLDERS) for lst in [FOLDERS, XTC_FILES, TPR_FILES, MM_LINK_INDEX_LISTS]), "The folder lists don't have an equal length"
    
def create_folder(folder_path, ask_overwrite=False):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    else:
        if ask_overwrite is True:
            answer = input(f"You are about to overwrite the folder {folder_path}. Are you sure? (y/n)")
            if answer=="y":
                shutil.rmtree(folder_path)
                os.mkdir(folder_path)
            else:
                print("Not overwritten. Aborting process")
                exit(1)

def clean_path(path):
    """Cleaning sometimes necessary, because absolute paths containing /srv/nfs don't work on the nodes
    """
    pattern = r"^/srv/nfs"
    return re.sub(pattern, '', path)
                
def run_single_point_calculation(geom_name, geom_filename, topol_filename, index_filename, singlepoint_files: dict):
    GROMACS_DFTB_PATH = os.path.abspath("/home/tkubar/GMX-DFTB/gromacs-dftbplus/release-machine-learning/bin/GMXRC")
    GROMACS_ORCA_PATH = os.path.abspath("/usr/local/run/gromacs-orca-avx2/bin/GMXRC")
    ORCA_PATH = os.path.abspath("/usr/local/run/orca_5_0_3_linux_x86-64_shared_openmpi411")
    ORCA_2MKL = os.path.abspath("/usr/local/run/orca_5_0_3_linux_x86-64_shared_openmpi411/orca_2mkl")
    BLAS_PATH = os.path.abspath("/usr/local/run/OpenBLAS-0.3.20-be/lib")
    PLUMED_PATH = os.path.abspath("/usr/local/run/plumed-2.5.1-openblas/lib")
    MULTIWFN_PATH = os.path.abspath("/home/lpetersen/Multiwfn/Multiwfn_3.8_dev_bin_Linux_noGUI/Multiwfn")
    job_file = "job_file.sge"
    
    mdp_filename = singlepoint_files["mdp"]
    orca_basename = singlepoint_files["orca_basename"]
    orcainfo_filename = singlepoint_files["orcainfo"]
    
    if index_filename is not None:
        index_command = f"-n {index_filename} "
    else:
        index_command = ""

    # Can't be indented for some things to work
    correct_environment_job_string=f'''#!/bin/bash
source_dir=$PWD
workdir=/scratch/$USER/gromacs_$JOB_ID
mkdir -p $workdir

cp {geom_filename} $workdir
cp {orcainfo_filename} $workdir/{orca_basename+".ORCAINFO"} # has to fit the GMX_QM_ORCA_BASENAME
cd $workdir


# single point calc with Orca
gmx_d grompp -f {mdp_filename} -c {geom_filename} -p {topol_filename} {index_command}-o sp.tpr
gmx_d mdrun -deffnm sp

# for charges obtained from ESP fitting
{ORCA_2MKL} $GMX_QM_ORCA_BASENAME -molden
echo -e "7\n12\n1\ny\n0\n0\nq" | {MULTIWFN_PATH} sp.molden.input > /dev/null

rm sp.densities
rm sp.gbw
rm sp.molden.input

cd $source_dir
rsync -a $workdir/ .
/bin/rm -rf $workdir
'''
   
    with open(job_file, "w") as f:
        f.write(correct_environment_job_string)
        
    return
    # Can't be indented for some things to work
    job_string = f'''#!/bin/bash
#$ -N {geom_name}
#$ -cwd
#$ -o {geom_name}.o$JOB_ID
#$ -e {geom_name}.e$JOB_ID

echo "JOB: $JOB_ID started on $HOSTNAME -- $(date)"
export LD_LIBRARY_PATH={BLAS_PATH}:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH={PLUMED_PATH}:$LD_LIBRARY_PATH
source_dir=$PWD
workdir=/scratch/$USER/gromacs_$JOB_ID
mkdir -p $workdir

cp {geom_filename} $workdir
cp {orcainfo_filename} $workdir/{orca_basename+".ORCAINFO"} # has to fit the GMX_QM_ORCA_BASENAME
cd $workdir
ulimit -s unlimited

# single point calc with Orca
source {GROMACS_ORCA_PATH}
export PATH={ORCA_PATH}:$PATH
export LD_LIBRARY_PATH={ORCA_PATH}:$LD_LIBRARY_PATH
export GMX_ORCA_PATH={ORCA_PATH}
export GMX_QMMM_VARIANT=2
export OMP_NUM_THREADS=1
export GMX_DFTB_ESP=1
export GMX_QM_ORCA_BASENAME={orca_basename}

gmx_d grompp -f {mdp_filename} -c {geom_filename} -p {topol_filename} {index_command}-o sp.tpr
gmx_d mdrun -deffnm sp

# for charges obtained from ESP fitting
{ORCA_2MKL} $GMX_QM_ORCA_BASENAME -molden
echo -e "7\n12\n1\ny\n0\n0\nq" | {MULTIWFN_PATH} sp.molden.input > /dev/null

rm sp.densities
rm sp.gbw
rm sp.molden.input

cd $source_dir
rsync -a $workdir/ .
/bin/rm -rf $workdir
'''
    with open(job_file, "w") as f:
        f.write(job_string)

    subprocess.run(f"qsub {job_file}", shell=True)
    
if __name__=="__main__":
    run_sp_calc_from_classic_traj(N_GEOMS, DATA_FOLDER, FOLDERS, XTC_FILES, TPR_FILES, MM_LINK_INDEX_LISTS, QM_LINK_INDEX_LISTS, SINGLEPOINT_FILES)