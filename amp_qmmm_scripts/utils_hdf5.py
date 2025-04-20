#!/usr/bin/env python
import argparse
from ase.io import read
import h5py
import json
import numpy as np
import os
import shutil
from typing import List, Tuple

# Unit conversions when constructing the HDF5 file from .extxyz files:
#   - Coordinates: [A] -> [A]
#   - Energies: [eV] -> [H]
#   - Forces: [eV/A] -> [H/a0]
#   - Gradients: [eV/A] -> [H/a0]
#   - Charges: [e] -> [e]
#   - Dipoles: [e*a0] -> [eA]
#   - Quadrupoles: [e*a0**2] -> [eA**2]

# Literature units:
# https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/707814/README.md
#   |- Identifier
#     |- orca_coordinates, Shape: (M, N, 3), [A]. Type: float64
#     |- orca_dipoles, Shape: (M, 3), [eA]. Type: float64
#     |- orca_energies, Shape: (M,), [H]. Type: float64
#     |- orca_engrad, Shape: (M, N, 3), [H/a0]. Type: float64
#     |- orca_pc_charges, Shape: (M, Z), [e]. Type: float64
#     |- orca_pc_coordinates, Shape: (M, Z, 3), [A]. Type: float64
#     |- orca_pcgrad, Shape: (M, Z, 3), [H/a0]. Type: float64
#     |- orca_quadrupoles, Shape: (M, 6), [eA**2]. Type: float64
#     |- orca_species, Shape: (M, N), Type: int64
#     |- xtb_coordinates, Shape: (M, N, 3), [a0]. Type: float64
#     |- xtb_energies, Shape: (M,), [H]. Type: float64
#     |- xtb_engrad, Shape: (M, N, 3), [H/a0]. Type: float64
#     |- xtb_pc_charges, Shape: (M, Z), [e]. Type: float64
#     |- xtb_pc_coordinates, Shape: (M, Z, 3), [a0]. Type: float64
#     |- xtb_pcgrad, Shape: (M, Z, 3), [H/a0]. Type: float64
#     |- xtb_species, Shape: (M, N), Type: int64
# ==> differences in xtb and orce coordinates: [A] vs [a0]

ORCA_CONVERSION_DICTIONARY = {
    'xtb_species': 'qm_charges', # Same for both
    'xtb_coordinates': 'qm_coordinates', # Same for both
    'orca_energies': 'qm_energies',
    'orca_engrad': 'qm_gradients',
    'orca_dipoles': 'qm_dipoles',
    'orca_quadrupoles': 'qm_quadrupoles',
    'xtb_pc_charges': 'mm_charges', # Same for both
    'xtb_pc_coordinates': 'mm_coordinates', # Same for both
    'orca_pcgrad': 'mm_gradients',
}

XTB_CONVERSION_DICTIONARY = {
    'xtb_species': 'qm_charges', # Same for both
    'xtb_coordinates': 'qm_coordinates', # Same for both
    'xtb_energies': 'qm_energies',
    'xtb_engrad': 'qm_gradients',
    'xtb_dipoles': 'qm_dipoles',
    'xtb_quadrupoles': 'qm_quadrupoles',
    'xtb_pc_charges': 'mm_charges', # Same for both
    'xtb_pc_coordinates': 'mm_coordinates', # Same for both
    'xtb_pcgrad': 'mm_gradients',
}

REDUNDANT_KEYS = {
    'orca_species': 'xtb_species',
    'orca_coordinates': 'xtb_coordinates',
    'orca_pc_charges': 'xtb_pc_charges',
    'orca_pc_coordinates': 'xtb_pc_coordinates',
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

# Expected like this in train_amp.py
TRAINING_DIRECTORY = "training"
VALIDATION_DIRECTORY = "validation"
TEST_DIRECTORY = "test"

SYSTEM_NAME = "dalanine"

eV_to_H = 27.211386245988
bohr_to_angstrom = 0.52917721067
angstrom_to_bohr = 1.0 / bohr_to_angstrom

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Utility script for handling HDF5 files and conversions.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-command to execute.")

    # Subparser for unpacking datasets from an HDF5 file
    unpack_parser = subparsers.add_parser("unpack", help="Unpack datasets from an HDF5 file and save them as .npy files.")
    unpack_parser.add_argument("hdf5_file_path", type=str, help="Path to the HDF5 file.")
    unpack_parser.add_argument("-o", "--output_dir", required=False, default=None, type=str, help="Output directory for .npy files.")
    unpack_parser.add_argument("-i", "--indices", nargs="+", type=int, default=None, help="Indices of the datasets to unpack.")
    unpack_parser.add_argument("-s", "--single_system", action="store_true", help="Unpack only the one dataset.")
    unpack_parser.add_argument("-n", "--name", type=str, default=None, help="Name of the system. Default is %s." % SYSTEM_NAME)
    unpack_parser.add_argument("--splits", nargs=3, type=float, default=None, help="Split each system into training, validation, and test sets. Provide the split ratios as three floats.")
    unpack_parser.add_argument("-v", "--view", action="store_true", help="View the structure of the HDF5 file.")
    unpack_parser.add_argument("-c", "--conversion", choices=["orca", "xtb"], default="orca", help="Conversion type: 'orca' or 'xtb'. Default is 'orca'.")

    # Subparser for creating an HDF5 file from extxyz and other files
    pack_parser = subparsers.add_parser("pack", help="Create an HDF5 file from extxyz and other files.")
    pack_parser.add_argument("hdf5_file_path", type=str, help="Path to the output HDF5 file.")
    pack_parser.add_argument("-e", "--extxyz", type=str, required=False, help="Path to the extxyz file for conversion to .hdf5.")
    pack_parser.add_argument("-o", "--output_dir", required=False, default=None, type=str, help="Output directory for the HDF5 file.")
    pack_parser.add_argument("-n", "--name", type=str, default=None, help="Name of the system. Default is %s." % SYSTEM_NAME)
    pack_parser.add_argument("--pc", type=str, default=None, help="Path to the concatenated pointcharges files. Optional")
    pack_parser.add_argument("--pcgrad", type=str, default=None, help="Path to the concatenated pointcharges gradient files. Optional")
    pack_parser.add_argument("-c", "--config", type=str, default=None, help="Path to the configuration file. Terminal commands have priority over this file.")

    concatenate_parser = subparsers.add_parser("concatenate", help="Concatenate multiple HDF5 files into one.")
    concatenate_parser.add_argument("hdf5_file_path", type=str, help="Path to the new HDF5 file.")
    concatenate_parser.add_argument("hdf5_file_paths", nargs="+", type=str, help="Paths to the HDF5 files to concatenate.") 
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
        if not args.view and not args.output_dir:
            args.view = True

        if args.single_system and args.indices is None and args.output_dir is not None:
            args.indices = [0]

        if args.single_system and len(args.indices) != 1 and args.output_dir is not None:
            parser.error("When --single_system is specified, exactly one index must be provided.")
        
        if not args.single_system and args.indices is None and args.output_dir is not None:
            parser.error("When --single_system is not specified, indices must be provided.")

        if args.splits is not None:
            if len(args.splits) != 3:
                parser.error("When --split is specified, exactly three ratios must be provided.")
            if sum(args.splits) > 1.0:
                parser.error("The sum of the split ratios must be smaller than or equal to 1.0.")
            if args.single_system:
                parser.warning("The --split option is ignored when --single_system is specified.")

    elif args.command == "pack":
        if args.extxyz is None:
            parser.error("The --extxyz option must be specified.")

        if not os.path.exists(args.extxyz):
            parser.error(f"The file {args.extxyz} does not exist.")
        args.extxyz = os.path.abspath(args.extxyz)
        
        if args.output_dir is not None:
            args.hdf5_file_path = os.path.join(args.output_dir, args.name + ".hdf5")
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
    prepare_output_directory(output_dir, None)

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
    output_dir = os.path.join(args.output_dir, args.name)
    indices: List[int] = args.indices
    splits: None | List[float] = args.splits

    print(f"Unpacking {hdf5_file_path} to {output_dir}")

    prepare_output_directory(output_dir, splits)

    hdf5_file = h5py.File(hdf5_file_path, "r")
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
        for key in group_dict.keys():
            if key in ORCA_CONVERSION_DICTIONARY.keys() and not key in ORCA_CONVERSION_DICTIONARY.values():
                raise ValueError(f"Unexpected key {key} in group {group_name}, expected one of {ORCA_CONVERSION_DICTIONARY.keys()} or {ORCA_CONVERSION_DICTIONARY.values()}")
        for key, value in DELTA_KEYS.items():
            high_order_key = value[0]
            low_order_key = value[1]
            if high_order_key in group_dict and low_order_key in group_dict:
                group_dict[key] = group_dict[high_order_key] - group_dict[low_order_key]
        group_dict = {ORCA_CONVERSION_DICTIONARY.get(key, key): value for key, value in group_dict.items()}

        if splits is None:
            # Save the group data as a .npy file
            np.save(os.path.join(output_dir, f"{group_name}_batch_{group_idx}.npy"), group_dict, allow_pickle=True)
            continue

        # Save the group data as a .npy file according to the split ratios
        first_group_value = list(group_dict.values())[0]
        n_group_entries = len(first_group_value)
        n_split_entries = [int(n_group_entries * split) for split in splits]
        random_indices_list = np.random.permutation(n_group_entries)
        split_starts = [0] + np.cumsum(n_split_entries).tolist()
        split_indices_list = [random_indices_list[split_starts[i]:split_starts[i + 1]] for i in range(len(split_starts) - 1)]
        
        for split_name, split_indices in zip([TRAINING_DIRECTORY, VALIDATION_DIRECTORY, TEST_DIRECTORY], split_indices_list):
            split_group_dict = {key: value[split_indices] for key, value in group_dict.items()}
            split_group_dir = os.path.join(output_dir, split_name)

            # Save the split group data as a .npy file
            np.save(os.path.join(split_group_dir, f"{group_name}_batch_{group_idx}.npy"), split_group_dict, allow_pickle=True)
            np.save(os.path.join(split_group_dir, f"{group_name}_indices_{group_idx}.npy"), split_indices)
            
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
        max_mm_molecules = find_max_chunk_size(point_charges_file_path)
        print(f"Max chunk size in {point_charges_file_path}: {max_mm_molecules}")
        assert max_mm_molecules == find_max_chunk_size(point_charges_grad_file_path), f"Max chunk size in {point_charges_file_path} and {point_charges_grad_file_path} differ."
        validate_chunk_sizes(point_charges_file_path, max_mm_molecules, threshold=100)

        point_charges_contents: List[np.ndarray] = read_chunked_file(point_charges_file_path) # Shape (n_mm_molecules, n_mm_atoms, 4), possibly irregular
        mm_gradients: List[np.ndarray]           = read_chunked_file(point_charges_grad_file_path) # Shape (n_mm_molecules, n_mm_atoms, 3), possibly irregular
        point_charges_contents = pad_arrays(point_charges_contents, max_mm_molecules)
        mm_gradients           = pad_arrays(mm_gradients, max_mm_molecules)
        mm_charges     = point_charges_contents[:, :, 0]
        mm_coordinates = point_charges_contents[:, :, 1:4]
    else:
        # Create dummy data for MM charges and coordinates
        mm_charges = np.zeros((len(molecules), 1))
        mm_coordinates = np.zeros((len(molecules), 1, 3))
        mm_gradients = np.zeros((len(molecules), 1, 3))

    assert mm_charges.shape[0] == len(molecules), f"MM charges shape {mm_charges.shape[0]} does not match number of molecules {len(molecules)}"
    assert mm_coordinates.shape[0] == len(molecules), f"MM coordinates shape {mm_coordinates.shape[0]} does not match number of molecules {len(molecules)}"
    assert mm_gradients.shape[0] == len(molecules), f"MM gradients shape {mm_gradients.shape[0]} does not match number of molecules {len(molecules)}"

    # unit conversion
    qm_energies = qm_energies*eV_to_H                           # ev -> H
    qm_forces = qm_forces*eV_to_H/angstrom_to_bohr              # eV/A -> H/a0
    qm_gradients = qm_forces*-1
    qm_dipoles = qm_dipoles*bohr_to_angstrom                    # e*a0 -> eA
    qm_quadrupoles = qm_quadrupoles*bohr_to_angstrom**2         # e*a0**2 -> eA**2

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
        array (numpy.ndarray): Array to pad
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

def prepare_output_directory(output_dir: str, splits: None | List[float]):
    """Prepare the output directory by cleaning up and creating necessary subdirectories."""
    # Remove the directory and its contents if it already exists
    if os.path.exists(output_dir) and os.path.isdir(output_dir) and not os.getcwd() == output_dir:
        shutil.rmtree(output_dir)
    elif os.getcwd() == output_dir:
        print(f"Output directory is the current working directory. Won't do cleanup.")

    os.makedirs(output_dir, exist_ok=True)

    if splits is not None:
        for split_group in [TRAINING_DIRECTORY, VALIDATION_DIRECTORY, TEST_DIRECTORY]:
            split_group_dir = os.path.join(output_dir, split_group)
            # Remove the directory and its contents if it already exists
            if os.path.exists(split_group_dir) and os.path.isdir(split_group_dir):
                shutil.rmtree(split_group_dir)
            os.makedirs(split_group_dir, exist_ok=True)

def main():
    """Main function to parse arguments and unpack HDF5 file."""
    args = parse_arguments()

    if args.command == "pack":
        # Create an .hdf5 file from an extxyz file
        pack_single_system(args)

    elif args.command == "unpack":
        if args.view:
            # View the structure of the HDF5 file
            view_hdf5_file(args.hdf5_file_path)
        elif args.output_dir and args.single_system:
            # Unpack a single system from the HDF5 file
            unpack_single_system(args)
        elif args.output_dir and not args.single_system:
            # Unpack multiple systems from the HDF5 file
            unpack_multiple_systems(args)

    elif args.command == "concatenate":
        # Concatenate multiple HDF5 files into one
        concatenate_hdf5_files(args.hdf5_file_path, args.hdf5_file_paths)

if __name__ == "__main__":
    main()