#!/usr/bin/env python
import argparse
from ase.io import read
import h5py
import json
import numpy as np
import os
import shutil
from typing import List, Tuple, Optional, Dict
import io

# Unit conversions when constructing the HDF5 file from .extxyz files:
#   - Coordinates: [A] -> [A]
#   - Energies: [eV] -> [H]
#   - Forces: [eV/A] -> [H/a0]
#   - Gradients: [eV/A] -> [H/a0]
#   - Charges: [e] -> [e]
#   - Dipoles: [Debye] -> [e*a0]
#   - Quadrupoles: [e*a0**2] -> [e*a0**2]

# Unit conversions when constructing the HDF5 file from .pc and .pcgrad files:
#   - Coordinates: [A] -> [A]
#   - Charges: [e] -> [e]
#   - Gradients: [H/a0] -> [H/a0]

# Literature units:
# https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/707814/README.md
#   |- Identifier
#     |- orca_coordinates, Shape: (M, N, 3), [A]. Type: float64
#     |- orca_dipoles, Shape: (M, 3), [eA]. Type: float64                <---------- Careful: Unit is wrong, it's [e*a0]
#     |- orca_energies, Shape: (M,), [H]. Type: float64
#     |- orca_engrad, Shape: (M, N, 3), [H/a0]. Type: float64
#     |- orca_pc_charges, Shape: (M, Z), [e]. Type: float64
#     |- orca_pc_coordinates, Shape: (M, Z, 3), [A]. Type: float64
#     |- orca_pcgrad, Shape: (M, Z, 3), [H/a0]. Type: float64
#     |- orca_quadrupoles, Shape: (M, 6), [eA**2]. Type: float64         <---------- Careful: Unit is wrong, it's [e*a0**2]
#     |- orca_species, Shape: (M, N), Type: int64
#     |- xtb_coordinates, Shape: (M, N, 3), [a0]. Type: float64          <---------- Careful: Coordinates in [a0]
#     |- xtb_energies, Shape: (M,), [H]. Type: float64
#     |- xtb_engrad, Shape: (M, N, 3), [H/a0]. Type: float64
#     |- xtb_pc_charges, Shape: (M, Z), [e]. Type: float64
#     |- xtb_pc_coordinates, Shape: (M, Z, 3), [a0]. Type: float64       <---------- Careful: Coordinates in [a0]
#     |- xtb_pcgrad, Shape: (M, Z, 3), [H/a0]. Type: float64
#     |- xtb_species, Shape: (M, N), Type: int64
# ==> differences in xtb and orce coordinates: [A] vs [a0]

ORCA_CONVERSION_DICTIONARY = {
    'xtb_species': 'qm_charges', # Same for both
    'orca_coordinates': 'qm_coordinates', # Not the same, units in [A] vs [a0]
    'orca_energies': 'qm_energies',
    'orca_engrad': 'qm_gradients',
    'orca_dipoles': 'qm_dipoles',
    'orca_quadrupoles': 'qm_quadrupoles',
    'xtb_pc_charges': 'mm_charges', # Same for both
    'orca_pc_coordinates': 'mm_coordinates', # Not the same, units in [A] vs [a0]
    'orca_pcgrad': 'mm_gradients',
}

# Do not simplay replace the CONVERSION_DICTIONARY with the ORCA one, because the unit conversion of the coordinates is different.
XTB_CONVERSION_DICTIONARY = {
    'xtb_species': 'qm_charges', # Same for both
    'xtb_energies': 'qm_energies',
    'xtb_engrad': 'qm_gradients',
    'xtb_dipoles': 'qm_dipoles',
    'xtb_quadrupoles': 'qm_quadrupoles',
    'xtb_pc_charges': 'mm_charges', # Same for both
    'xtb_pcgrad': 'mm_gradients',
}

REDUNDANT_KEYS = {
    'orca_species': 'xtb_species',
    'orca_pc_charges': 'xtb_pc_charges',
}

DANGEROUS_REDUNDANT_KEYS = {
    'orca_coordinates': 'xtb_coordinates', # Not the same, units in [A] vs [a0]
    'orca_pc_coordinates': 'xtb_pc_coordinates', # Not the same, units in [A] vs [a0]
}

DELTA_KEYS = {
    'delta_qm_energies': ['orca_energies', 'xtb_energies'],
    'delta_qm_gradients': ['orca_engrad', 'xtb_engrad'],
    'delta_mm_gradients': ['orca_pcgrad', 'xtb_pcgrad'],
} # orca_property - xtb_property

EXTXYZ_KEYS = {
    "energy": "ref_energy",
    "forces": "ref_force",
    "dipole": "ref_dipole",
    "quadrupole": "ref_quadrupole"
}

# EXTXYZ_KEYS = {
#     "energy": "pred_energy",
#     "forces": "pred_force",
#     "dipole": "pred_dipole",
#     "quadrupole": "pred_quadrupole"
# }

# Expected like this in train_amp.py
TRAINING_DIRECTORY = "training"
VALIDATION_DIRECTORY = "validation"
TEST_DIRECTORY = "test"

OUTPUTDIR = os.getcwd()
SYSTEM_NAME = "dalanine"

BATCH_SIZE = 1000  # Default batch size for unpacking

H_to_eV = 27.211386245988
eV_to_H = 1.0 / H_to_eV
bohr_to_angstrom = 0.52917721067
angstrom_to_bohr = 1.0 / bohr_to_angstrom
H_B_to_ev_A = H_to_eV / bohr_to_angstrom
ev_A_to_H_B = 1.0 / H_B_to_ev_A
debye_to_ea0 = 0.3934303

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Utility script for handling HDF5 files and conversions.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-command to execute.")

    # Subparser for unpacking datasets from an HDF5 file
    unpack_parser = subparsers.add_parser("unpack", help="Unpack datasets from an HDF5 file and save them as .npy files.")
    unpack_parser.add_argument("hdf5_file_path", type=str, help="Path to the HDF5 file.")
    unpack_parser.add_argument("-o", "--output_dir", required=False, default=OUTPUTDIR, type=str, help="Output directory for .npy files. Default is the current directory.")
    unpack_parser.add_argument("-i", "--indices", nargs="+", type=int, default=None, help="Indices of the datasets to unpack(space-separated integers). Default is all datasets/first dataset if --single_system is specified.")
    unpack_parser.add_argument("-s", "--single_system", action="store_true", help="Unpack only the one dataset.")
    unpack_parser.add_argument("-n", "--name", type=str, default=None, help="Name of the system. Default is %s." % SYSTEM_NAME)
    unpack_parser.add_argument("--splits", nargs=3, type=float, default=None, help="Split each system into training, validation, and test sets. Provide the split ratios as three floats.")
    unpack_parser.add_argument("--n_splits", type=int, default=1, help="Number of train-valid-test splits. Default is 1.")
    unpack_parser.add_argument("-c", "--conversion", choices=["orca", "xtb"], default="orca", help="Conversion type: 'orca' or 'xtb'. Default is 'orca'.")

    # Subparser for creating an HDF5 file from extxyz and other files
    pack_parser = subparsers.add_parser("pack", help="Create an HDF5 file from extxyz and other files.")
    pack_parser.add_argument("hdf5_file_path", type=str, help="Path to the output HDF5 file.")
    pack_parser.add_argument("-e", "--extxyz", type=str, required=False, help="Path to the extxyz file for conversion to .hdf5.")
    pack_parser.add_argument("-o", "--output_dir", required=False, default=OUTPUTDIR, type=str, help="Output directory for the HDF5 file. Default is the current directory.")
    pack_parser.add_argument("-n", "--name", type=str, default=None, help="Name of the system. Default is %s." % SYSTEM_NAME)
    pack_parser.add_argument("--pc", type=str, default=None, help="Path to the concatenated pointcharges files. Optional")
    pack_parser.add_argument("--pcgrad", type=str, default=None, help="Path to the concatenated pointcharges gradient files. Optional")
    pack_parser.add_argument("-c", "--config", type=str, default=None, help="Path to the configuration file. Terminal commands have priority over this file.")

    # Subparser for concatenating multiple HDF5 files into one
    concatenate_parser = subparsers.add_parser("concatenate", help="Concatenate multiple HDF5 files into one.")
    concatenate_parser.add_argument("hdf5_file_path", type=str, help="Path to the new HDF5 file.")
    concatenate_parser.add_argument("hdf5_file_paths", nargs="+", type=str, help="Paths to the HDF5 files to concatenate.")

    # Subparser for viewing the structure of an HDF5 file
    view_parser = subparsers.add_parser("view", help="View the structure of an HDF5 file.")
    view_parser.add_argument("hdf5_file_path", type=str, help="Path to the HDF5 file.")
    args = parser.parse_args()

    # Check if the config file is provided
    if hasattr(args, "config") and args.config is not None:
        if os.path.exists(args.config):
            with open(args.config, "r") as f:
                config = json.load(f)
            for key, value in config.items():
                if hasattr(args, key):
                    setattr(args, key, value)
                else:
                    print(f"Warning: {key} in {args.config} is not a valid argument.")

    if args.command == "unpack":
        if not os.path.exists(args.hdf5_file_path):
            parser.error(f"The file {args.hdf5_file_path} does not exist.")
        elif not args.hdf5_file_path.endswith(".h5") and not args.hdf5_file_path.endswith(".hdf5"):
            parser.error(f"The file {args.hdf5_file_path} is not a valid HDF5 file.")

    if hasattr(args, "name") and args.name is None:
        args.name = SYSTEM_NAME

    if hasattr(args, "output_dir") and args.output_dir is not None:
        args.output_dir = os.path.abspath(args.output_dir)
    
    if args.command == "unpack":
        if args.single_system and args.indices is None:
            args.indices = [0]

        if args.single_system and len(args.indices) != 1:
            parser.error("When --single_system is specified, exactly one index must be provided.")
        
        if args.splits is not None:
            if len(args.splits) != 3:
                parser.error("When --split is specified, exactly three ratios must be provided.")
            if sum(args.splits) > 1.0:
                parser.error("The sum of the split ratios must be smaller than or equal to 1.0.")
            if args.single_system:
                print("The --split option is ignored when --single_system is specified.")

        if args.n_splits < 1:
            parser.error("The --n_splits option must be at least 1.")

        if args.n_splits > 1 and args.single_system:
            parser.error("The --n_splits option cannot be used with --single_system.")

    elif args.command == "pack":
        if args.extxyz is None:
            parser.error("The --extxyz option must be specified.")

        if not os.path.exists(args.extxyz):
            parser.error(f"The file {args.extxyz} does not exist.")
        args.extxyz = os.path.abspath(args.extxyz)
        
        if args.output_dir is not None:
            args.hdf5_file_path = os.path.join(args.output_dir, args.hdf5_file_path)
            args.output_dir = None
        
        if args.pc is not None and args.pcgrad is None:
            parser.error("When --pc is specified, --pcgrad must also be specified.")
        if args.pcgrad is not None and args.pc is None:
            parser.error("When --pcgrad is specified, --pc must also be specified.")

    elif args.command == "concatenate":
        if len(args.hdf5_file_paths) < 2:
            parser.error("At least two HDF5 files must be provided for concatenation.")
        for hdf5_file_path in args.hdf5_file_paths:
            if not os.path.exists(hdf5_file_path):
                parser.error(f"The file {hdf5_file_path} does not exist.")
            elif not hdf5_file_path.endswith(".h5") and not hdf5_file_path.endswith(".hdf5"):
                parser.error(f"The file {hdf5_file_path} is not a valid HDF5 file.")

    if args.command == "unpack" and args.conversion != "orca":
        raise NotImplementedError(f"Conversion type {args.conversion} is not implemented. Only 'orca' is supported.") 

    # Print the parsed arguments
    print("Parsed arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    return args

def view_hdf5_file(hdf5_file_path: str) -> None:
    def print_hdf5_structure(name, obj):
        if isinstance(obj, h5py.Group):
            print(f"Group: {name}")
        elif isinstance(obj, h5py.Dataset):
            print(f"  Dataset: {name}, Shape: {obj.shape}, Type: {obj.dtype}")

    with h5py.File(hdf5_file_path, "r") as hdf5_file:
        hdf5_file.visititems(print_hdf5_structure)

def unpack_single_system(args: argparse.Namespace) -> None:
    """Unpack datasets from an HDF5 file and save them as .npy files."""
    hdf5_file_path = args.hdf5_file_path
    output_dir = os.path.join(args.output_dir, args.name)
    index: int = args.indices[0]
    
    print(f"Unpacking {hdf5_file_path} to {output_dir}")

    # Prepare the output directory
    prepare_output_directory(output_dir, splits=None)

    #datasets: Dict[str, List[np.ndarray]] = {}
    hdf5_file = h5py.File(hdf5_file_path, "r")
    
    group_names = list(hdf5_file.keys())
    assert index < len(group_names), f"Index {index} out of range for group names. Amount of groups: {len(group_names)}"
    group_name = group_names[index]
    group = hdf5_file[group_name]

    if not isinstance(group, h5py.Group):
        raise ValueError(f"Expected group, but got {type(group)} for {group_name}")

    # Unpack datasets in the group
    for dataset_name in group.keys():
        dataset = group[dataset_name]
        if not isinstance(dataset, h5py.Dataset):
            print(f"Expected dataset, but got {type(dataset)} for {dataset_name}")
            continue

        #if dataset_name not in datasets:
            #datasets[dataset_name] = []

        print(f"Dataset: {dataset_name}, Shape: {dataset.shape}, Type: {dataset.dtype}")
        np.save(os.path.join(output_dir, f"{dataset_name}.npy"), dataset)
        #datasets[dataset_name].append(dataset)
            
    hdf5_file.close()
    
    # Create redundant files
    for key, value in REDUNDANT_KEYS.items():
        if os.path.exists(os.path.join(output_dir, f"{value}.npy")):
            shutil.copy(os.path.join(output_dir, f"{value}.npy"), os.path.join(output_dir, f"{key}.npy"))
    
    for key, value in DANGEROUS_REDUNDANT_KEYS.items():
        if os.path.exists(os.path.join(output_dir, f"{value}.npy")) and not os.path.exists(os.path.join(output_dir, f"{key}.npy")):
            redundant_data = np.load(os.path.join(output_dir, f"{value}.npy"), allow_pickle=False)
            redundant_data = redundant_data*bohr_to_angstrom
            np.save(os.path.join(output_dir, f"{key}.npy"), redundant_data)

    #INFO: Doesn't work, because the amount MM particles differs 
    # datasets = {dataset_name: np.concatenate(dataset) for dataset_name, dataset in datasets.items()}
    # # Save each dataset as a .npy file
    # for dataset_name, dataset in datasets.items():
    #     output_file_path = os.path.join(output_dir, f"{dataset_name}.npy")
    #     np.save(output_file_path, dataset)
    #     print(f"Saved {dataset_name} to {output_file_path}")

def unpack_multiple_systems(args: argparse.Namespace) -> None:
    """Unpack datasets from an HDF5 file for multiple systems and save them as .npy files."""
    hdf5_file_path = args.hdf5_file_path
    main_output_dir = args.output_dir
    indices: Optional[List[int]] = args.indices
    n_splits: int = args.n_splits
    splits: Optional[List[float]] = args.splits

    print(f"Unpacking {hdf5_file_path} to {main_output_dir}")

    output_dirs: List[str] = []
    if n_splits == 1:
        output_dir = os.path.join(main_output_dir, args.name)
        prepare_output_directory(output_dir, splits=splits)
        output_dirs.append(output_dir)
    else:
        for split_idx in range(n_splits):
            output_dir = os.path.join(main_output_dir, "splits", f"split_{split_idx}", args.name)
            prepare_output_directory(output_dir, splits=splits)
            output_dirs.append(output_dir)

    hdf5_file = h5py.File(hdf5_file_path, "r")

    if indices is None:
        # If no indices are provided, unpack all groups
        indices = list(range(len(hdf5_file.keys())))
        print(f"No indices provided. Unpacking all {len(indices)} groups.")

    # Iterate through all groups in the HDF5 file
    for group_idx, group_name in enumerate(hdf5_file.keys()):
        if group_idx > max(indices):
            print(f"Reached the last index {max(indices)}. Stopping unpacking.")
            break

        if group_idx not in indices:
            continue

        group = hdf5_file[group_name]
        if not isinstance(group, h5py.Group):
            print(f"Expected group, but got {type(group)} for {group_name}")
            continue
        
        group_dict = {key: np.array(value) for key, value in group.items() if isinstance(value, h5py.Dataset)}
        for key, value in DELTA_KEYS.items():
            high_order_key = value[0]
            low_order_key = value[1]
            if high_order_key in group_dict and low_order_key in group_dict:
                group_dict[key] = group_dict[high_order_key] - group_dict[low_order_key]

        # Convert units for the redundant xtb keys
        for key, value in DANGEROUS_REDUNDANT_KEYS.items():
            if value in group_dict and key not in group_dict:
                group_dict[key] = group_dict[value] * bohr_to_angstrom
                
        converted_group_dict = {ORCA_CONVERSION_DICTIONARY.get(key, key): value for key, value in group_dict.items()} # Rename keys according to the conversion dictionary

        for key in ORCA_CONVERSION_DICTIONARY.values():
            assert key in converted_group_dict.keys(), f"Key {key} not found in converted group dictionary for {group_name}. Available keys: {converted_group_dict.keys()}"

        if splits is None:
            save_numpy_batches(main_output_dir, group_name, converted_group_dict)
            continue

        # Save the group data as a .npy file according to the split ratios
        first_group_value = list(converted_group_dict.values())[0]
        n_group_entries = len(first_group_value)
        n_split_entries = [int(n_group_entries * split) for split in splits]
        remainder = n_group_entries - sum(n_split_entries)
        for i in range(remainder):
            n_split_entries[i % len(n_split_entries)] += 1
        # Shuffle all indices, then split them into train/val/test according to n_split_entries
        random_indices_list = np.random.permutation(n_group_entries)
        split_starts = [0] + np.cumsum(n_split_entries).tolist()
        split_indices_list = [random_indices_list[split_starts[i]:split_starts[i + 1]] for i in range(len(split_starts) - 1)]
        
        split_names = [TRAINING_DIRECTORY, VALIDATION_DIRECTORY, TEST_DIRECTORY]
        for split_name, split_indices in zip(split_names, split_indices_list): # Iterate over training, validation, and test splits
            for split_idx, output_dir in enumerate(output_dirs): #  Iterate over main_outputdir or split_0, split_1, split_2, etc.
                if split_name != TEST_DIRECTORY: 
                    split_split_indices = split_indices[split_idx::len(output_dirs)] # Split indices for the current split, take every n-th index where n is the number of output directories
                else:
                    split_split_indices = split_indices # For test split, take all indices to have same test set for all splits
                split_group_dict = {key: value[split_split_indices] for key, value in converted_group_dict.items()}
                split_group_dir = os.path.join(output_dir, split_name)
                os.makedirs(split_group_dir, exist_ok=True)
                save_numpy_batches(split_group_dir, group_name, split_group_dict, split_split_indices)
            
    hdf5_file.close()

    if max(indices) > group_idx:
        print(f"Warning: The indices {indices} are larger than the number of groups in the HDF5 file. Only unpacked up to index {group_idx}.")

    return

def pack_single_system(args: argparse.Namespace) -> None:
    """Pack a single system from extxyz to HDF5."""
    extxyz_file_path: str = args.extxyz
    point_charges_file_path: str|None = args.pc
    point_charges_grad_file_path: str|None = args.pcgrad
    hdf5_file_path: str = args.hdf5_file_path

    print(f"Packing {extxyz_file_path}, {point_charges_file_path}, {point_charges_grad_file_path} to {hdf5_file_path}")

    # Collect the data from the extxyz file
    molecules = read(extxyz_file_path, index=":")

    qm_charges, qm_coordinates, qm_energies, qm_forces, qm_dipoles, qm_quadrupoles = [], [], [], [], [], []
    for idx, molecule in enumerate(molecules):
        try:
            qm_charges.append(molecule.get_atomic_numbers())
            qm_coordinates.append(molecule.get_positions())
            if EXTXYZ_KEYS["energy"] == "energy": # ase uses this key for itself since a certain version
                qm_energies.append(molecule.get_potential_energy())
            else:
                qm_energies.append(molecule.info[EXTXYZ_KEYS["energy"]])
            if EXTXYZ_KEYS["forces"] == "forces": # ase uses this key for itself since a certain version
                qm_forces.append(molecule.get_forces())
            else:
                qm_forces.append(molecule.arrays[EXTXYZ_KEYS["forces"]])
            if EXTXYZ_KEYS["dipole"] == "dipole": # ase uses this key for itself since a certain versions
                qm_dipoles.append(molecule.get_dipole_moment())
            else:
                qm_dipoles.append(molecule.info[EXTXYZ_KEYS["dipole"]])
            qm_quadrupoles.append(molecule.info[EXTXYZ_KEYS["quadrupole"]])
        except KeyError as e:
            print(f"KeyError for molecule {idx}: {e}")
            print(f"Available keys: {molecule.info.keys()}")
            raise
        except Exception as e:
            print(f"Unexpected error for molecule {idx}: {e}")
            print(f"Available operations: {dir(molecule)}")
            raise

    qm_charges = np.array(qm_charges) # Shape (n_molecules, n_qm_atoms)
    qm_coordinates = np.array(qm_coordinates) # Shape (n_molecules, n_qm_atoms, 3)
    qm_energies = np.array(qm_energies).squeeze() # Shape (n_molecules,)
    qm_forces = np.array(qm_forces) # Shape (n_molecules, n_qm_atoms, 3)
    qm_dipoles = np.array(qm_dipoles) # Shape (n_molecules, 3)
    qm_quadrupoles = np.array(qm_quadrupoles) # Shape (n_molecules, 6)

    if point_charges_file_path is not None and point_charges_grad_file_path is not None:
        # Collect the data from the point charges files
        mm_charges, mm_coordinates, mm_gradients = extract_mm_data(point_charges_file_path, point_charges_grad_file_path)
    elif point_charges_file_path is None and point_charges_grad_file_path is None:
        # Create dummy data for MM charges and coordinates
        mm_charges = np.zeros((len(molecules), 1))
        mm_coordinates = np.zeros((len(molecules), 1, 3)) + 0.01 # Avoid zero coordinates, which are otherwise used for padding and removed
        mm_gradients = np.zeros((len(molecules), 1, 3))
    else:
        raise ValueError("Both --pc and --pcgrad must be specified or both must be None.")

    assert mm_charges.shape[0] == len(molecules), f"MM charges shape {mm_charges.shape[0]} does not match number of molecules {len(molecules)}"
    assert mm_coordinates.shape[0] == len(molecules), f"MM coordinates shape {mm_coordinates.shape[0]} does not match number of molecules {len(molecules)}"
    assert mm_gradients.shape[0] == len(molecules), f"MM gradients shape {mm_gradients.shape[0]} does not match number of molecules {len(molecules)}"

    # unit conversion
    qm_energies = qm_energies*eV_to_H                           # ev -> H
    qm_forces = qm_forces*ev_A_to_H_B                           # eV/A -> H/a0
    qm_gradients = qm_forces*-1
    qm_dipoles = qm_dipoles*debye_to_ea0                        # Debye -> e*a0
    qm_quadrupoles = qm_quadrupoles*1**2                        # e*a0**2 -> e*a0**2

    # Convert the molecules to a dictionary and 
    molecules_dict = {
        'qm_charges': qm_charges,
        'qm_coordinates': qm_coordinates,
        'qm_energies': qm_energies,
        'qm_forces': qm_forces,
        'qm_gradients': qm_gradients,
        'qm_dipoles': qm_dipoles,
        'qm_quadrupoles': qm_quadrupoles,
        'mm_charges': mm_charges,
        'mm_coordinates': mm_coordinates,
        'mm_gradients': mm_gradients,
    }

    # Create a new HDF5 file
    with h5py.File(hdf5_file_path, "a") as hdf5_file:
        group_name = args.name
        if group_name in hdf5_file:
            print(f"Group {group_name} already exists in {hdf5_file_path}. Overwriting.")
            del hdf5_file[group_name]

        group = hdf5_file.create_group(group_name)

        # Save the data to the HDF5 file
        for key, value in molecules_dict.items():
            group.create_dataset(key, data=value)

def concatenate_hdf5_files(new_hdf5_file_path: str, hdf5_file_paths: List[str]) -> None:
    """
    Concatenate multiple HDF5 files into one, renaming groups to prevent conflicts.
    
    Args:
        new_hdf5_file_path (str): Path to the new HDF5 file to create
        hdf5_file_paths (List[str]): List of paths to the HDF5 files to concatenate
    """
    print(f"Concatenating {len(hdf5_file_paths)} HDF5 files into {new_hdf5_file_path}")
    
    # Create the new HDF5 file
    with h5py.File(new_hdf5_file_path, "w") as new_file:
        group_counter = 0
        
        # Process each source file
        for source_path in hdf5_file_paths:
            print(f"Processing file: {source_path}")
            
            with h5py.File(source_path, "r") as source_file:
                # Copy each group with a new index
                for group_name in source_file.keys():
                    # Create a new name with an incremented index
                    new_group_name = f"{group_name}_{group_counter:03}"
                    print(f"  Copying group {group_name} as {new_group_name}")
                    
                    # Create the new group
                    new_group = new_file.create_group(new_group_name)
                    
                    # Copy all datasets from the source group to the new group
                    source_group = source_file[group_name]
                    for dataset_name, dataset in source_group.items():
                        if isinstance(dataset, h5py.Dataset):
                            new_group.create_dataset(dataset_name, data=dataset[:])
                    
                    group_counter += 1
    
    print(f"Successfully concatenated {group_counter} groups into {new_hdf5_file_path}")

def find_max_chunk_size(filename: str) -> int:
    """
    Find the maximum chunk size in a file where each chunk starts with a size indicator.
    
    Args:
        filename (str): Path to the file to analyze
        
    Returns:
        int: Maximum chunk size found in the file
    """
    max_size = 0
    
    with open(filename, 'r') as file:
        while True:
            # Try to read the chunk size
            size_line = file.readline().strip()
            if not size_line:  # End of file
                break
                
            try:
                chunk_size = int(size_line)
                max_size = max(max_size, chunk_size)
                
                # Skip the chunk data to get to the next size indicator
                for _ in range(chunk_size):
                    if not file.readline():  # Unexpected end of file
                        break
                        
            except ValueError:
                # Not a size indicator line, skip
                continue
    
    return max_size

def validate_chunk_sizes(filename: str, expected_max_size: int, threshold: int = 100) -> List[Tuple[int, int]]:
    """
    Validate chunk sizes in a file against an expected maximum size and identify outliers.
    
    Args:
        filename (str): Path to the file to analyze
        expected_max_size (int): Expected maximum chunk size to check against
        threshold (int): Threshold for determining significant differences (default: 100)
        
    Returns:
        List[Tuple[int, int]]: List of tuples (chunk_index, chunk_size) for chunks with significant size differences
    """
    outliers = []
    chunk_index = 0
    
    with open(filename, 'r') as file:
        while True:
            # Try to read the chunk size
            size_line = file.readline().strip()
            if not size_line:  # End of file
                break

            try:
                chunk_size = int(size_line)
            except ValueError:
                # Not a size indicator line, skip
                continue


            
            # Check if the chunk size differs significantly from the expected max
            if abs(chunk_size - expected_max_size) >= threshold:
                outliers.append((chunk_index, chunk_size))
                print(f"Warning: Chunk {chunk_index} has size {chunk_size}, which differs "
                        f"from expected max size {expected_max_size} by {abs(chunk_size - expected_max_size)}")
            
            # Skip the chunk data to get to the next size indicator
            for _ in range(chunk_size):
                if not file.readline():  # Unexpected end of file
                    break
            
            chunk_index += 1
                        
    
    if outliers:
        outlier_percentage = (len(outliers) / chunk_index) * 100
        print(f"Found {len(outliers)} outliers out of {chunk_index} chunks ({outlier_percentage:.2f}%)")
        print("This may bias the training of the model, because missing data gets filled with zeros.")
        print("Consider creating separate systems for different sizes.")
    else:
        print(f"All {chunk_index} chunks are within {threshold} of the expected maximum size {expected_max_size}")
        
    return outliers

def read_chunked_file(filename: str) -> List[np.ndarray]:
    """
    Read a file in chunks where each chunk starts with a size indicator.
    Returns a list of numpy arrays for each chunk based on the specified size.
    
    Args:
        filename (str): Path to the file to read
        
    Returns:
        List[numpy.ndarray]: List of arrays, each with shape (chunk_size, *) for each chunk
    """
    result = []
    with open(filename, 'r') as file:
        while True:
            # Try to read the chunk size
            size_line = file.readline().strip()
            if not size_line:  # End of file
                break
                
            try:
                chunk_size = int(size_line)
            except ValueError:
                # Not a size indicator line, skip
                continue
                
            # Read the specified number of lines
            chunk_data = []
            for _ in range(chunk_size):
                line = file.readline()
                if not line:  # Unexpected end of file
                    break
                    
                values = line.strip().split()
                chunk_data.append(values)

            if chunk_data:
                result.append(np.array(chunk_data, dtype=float))

    return result

def pad_arrays(arrays: List[np.ndarray], target_length: int) -> np.ndarray:
    """
    Pad numpy arrays to a target length with zeros along their current axis 0. Expected shape is (n_mm_atoms, m).
    
    Args:
        arrays (List[numpy.ndarray]): List of arrays to pad
        target_length (int): Target length for padding
        
    Returns:
        numpy.ndarray: Stacked arrays along axis 0, padded to the target length. Return shape (n_mm_molecules, max_n_mm_atoms, m)
    """
    padded_arrays = []
    for array in arrays:
        pad_width = [(0, target_length - array.shape[0])] + [(0, 0)] * (len(array.shape) - 1)
        padded_array = np.pad(array, pad_width, mode='constant', constant_values=0)
        padded_arrays.append(padded_array)
    
    return np.stack(padded_arrays, axis=0)

def extract_mm_data(point_charges_file_path: str, point_charges_grad_file_path: Optional[str]=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Handle the MM files for point charges and gradients, reading them in chunks and padding them to a consistent size.

    Args:
        point_charges_file_path (str): path to the point charges file
        point_charges_grad_file_path (str): path to the point charges gradient file

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: mm_charges, mm_coordinates, mm_gradients
    """
    max_mm_molecules = find_max_chunk_size(point_charges_file_path)
    print(f"Max chunk size in {point_charges_file_path}: {max_mm_molecules}")
    validate_chunk_sizes(point_charges_file_path, max_mm_molecules, threshold=100)

    point_charges_contents: List[np.ndarray] = read_chunked_file(point_charges_file_path) # Shape (n_mm_molecules, n_mm_atoms, 4), possibly irregular
    point_charges_contents = pad_arrays(point_charges_contents, max_mm_molecules)
    mm_charges     = point_charges_contents[:, :, 0]
    mm_coordinates = point_charges_contents[:, :, 1:4]
    
    if point_charges_grad_file_path is None:
        # If no gradients file is provided, create dummy gradients
        mm_gradients = np.zeros_like(mm_coordinates)
    else:
        assert max_mm_molecules == find_max_chunk_size(point_charges_grad_file_path), f"Max chunk size in {point_charges_file_path} and {point_charges_grad_file_path} differ."
        mm_gradients: List[np.ndarray] = read_chunked_file(point_charges_grad_file_path) # Shape (n_mm_molecules, n_mm_atoms, 3), possibly irregular
        mm_gradients = pad_arrays(mm_gradients, max_mm_molecules)

    return mm_charges, mm_coordinates, mm_gradients

def prepare_output_directory(output_dir: str, splits: Optional[List[float]] = None) -> None:
    """Prepare the output directory by cleaning up and creating necessary subdirectories."""
    # Remove the directory and its contents if it already exists
    if os.path.exists(output_dir) and os.path.isdir(output_dir) and not os.getcwd() == output_dir:
        shutil.rmtree(output_dir)
    elif os.getcwd() == output_dir:
        print(f"Output directory is the current working directory. Won't do cleanup.")

    os.makedirs(output_dir, exist_ok=True)

    # Create subdirectories for training, validation, and test sets if splits are provided
    if splits is not None:
        for split_group in [TRAINING_DIRECTORY, VALIDATION_DIRECTORY, TEST_DIRECTORY]:
            split_group_dir = os.path.join(output_dir, split_group)
            # Remove the directory and its contents if it already exists
            if os.path.exists(split_group_dir) and os.path.isdir(split_group_dir):
                shutil.rmtree(split_group_dir)
            os.makedirs(split_group_dir, exist_ok=True)

def save_numpy_batches(output_dir: str, group_name: str, data_dict: Dict[str, np.ndarray], split_indices: Optional[np.ndarray] = None) -> None:
    """Save a batch of numpy arrays to the output directory."""
    n_entries = len(next(iter(data_dict.values())))
    n_batches = (n_entries + BATCH_SIZE - 1) // BATCH_SIZE  # Calculate number of batches needed


    for batch_idx in range(n_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, n_entries)
        
        # Create a new dictionary for the current batch
        if end_idx > 1:
            batch_data_dict = {key: value[start_idx:end_idx] for key, value in data_dict.items()}
        else:
            batch_data_dict = data_dict  # If only one entry, use the full data_dict

        # Check the byte size of the converted_group_dict before saving, numpy can only save arrays smaller than 4.0 GB
        if batch_idx == 0:
            buffer = io.BytesIO()
            np.save(buffer, batch_data_dict, allow_pickle=True)
            byte_size = buffer.tell()
            assert byte_size < 3.9*1024*1024*1024, f"Batch {group_name} exceeds 4.0 GB limit: {byte_size / (1024 * 1024 * 1024):.2f} GB. Adjust BATCH_SIZE."
        
        # Save the batch data as a .npy file
        batch_file_path = os.path.join(output_dir, f"{group_name}_batch_{batch_idx}.npy")
        np.save(batch_file_path, batch_data_dict, allow_pickle=True)
        
        if split_indices is not None:
            # Save the indices for the current batch
            indices_file_path = os.path.join(output_dir, f"{group_name}_indices_{batch_idx}.npy")
            np.save(indices_file_path, split_indices[start_idx:end_idx])

def main():
    """Main function to parse arguments and unpack HDF5 file."""
    args = parse_arguments()

    if args.command == "pack":
        # Create an .hdf5 file from an extxyz file
        pack_single_system(args)

    elif args.command == "unpack":
        if args.output_dir and args.single_system:
            # Unpack a single system from the HDF5 file
            unpack_single_system(args)
        elif args.output_dir and not args.single_system:
            # Unpack multiple systems from the HDF5 file
            unpack_multiple_systems(args)

    elif args.command == "concatenate":
        # Concatenate multiple HDF5 files into one
        concatenate_hdf5_files(args.hdf5_file_path, args.hdf5_file_paths)

    elif args.command == "view":
        # View the structure of an HDF5 file
        view_hdf5_file(args.hdf5_file_path)

    else:
        print("Invalid command. Use 'unpack', 'pack', 'concatenate', or 'view'.")

if __name__ == "__main__":
    main()