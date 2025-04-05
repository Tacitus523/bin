#!/usr/bin/env python
import h5py
import numpy as np
import os
import argparse
import shutil
from typing import Dict, List

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

DELTA_KEYS = {
    'delta_qm_energies': ['orca_energies', 'xtb_energies'],
    'delta_qm_gradients': ['orca_engrad', 'xtb_engrad'],
    'delta_mm_gradients': ['orca_pcgrad', 'xtb_pcgrad'],
} # orca_property - xtb_property

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Unpack datasets from an HDF5 file and save them as .npy files.")
    parser.add_argument("hdf5_file_path", type=str, help="Path to the HDF5 file.")
    parser.add_argument("-o", "--output_dir", required=False, default=None, type=str, help="Output directory for .npy files.")
    parser.add_argument("-v", "--view", action="store_true", help="View the structure of the HDF5 file.")
    parser.add_argument("-i", "--indices", nargs="+", type=int, default=None, help="Indices of the datasets to unpack.")
    parser.add_argument("-s", "--single_system", action="store_true", help="Unpack only the one dataset.")
    args = parser.parse_args()

    if not os.path.exists(args.hdf5_file_path):
        parser.error(f"The file {args.hdf5_file_path} does not exist.")
    
    if not args.view and not args.output_dir:
        args.view = True

    if args.single_system and args.indices is None and args.output_dir is not None:
        args.indices = [0]

    if args.single_system and len(args.indices) != 1 and args.output_dir is not None:
        parser.error("When --single_system is specified, exactly one index must be provided.")
    
    if not args.single_system and args.indices is None and args.output_dir is not None:
        parser.error("When --single_system is not specified, indices must be provided.")

    return args

def view_hdf5_file(hdf5_file_path):
    def print_hdf5_structure(name, obj):
        if isinstance(obj, h5py.Group):
            print(f"Group: {name}")
        elif isinstance(obj, h5py.Dataset):
            print(f"  Dataset: {name}, Shape: {obj.shape}, Type: {obj.dtype}")

    with h5py.File(hdf5_file_path, "r") as hdf5_file:
        hdf5_file.visititems(print_hdf5_structure)

def unpack_single_system(hdf5_file_path, output_dir, index: int):
    """Unpack datasets from an HDF5 file and save them as .npy files."""
    print(f"Unpacking {hdf5_file_path} to {output_dir}")

    # Remove the directory and its contents if it already exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

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
    
    #INFO: Doesn't work, because the amount MM particles differs 
    # datasets = {dataset_name: np.concatenate(dataset) for dataset_name, dataset in datasets.items()}
    # # Save each dataset as a .npy file
    # for dataset_name, dataset in datasets.items():
    #     output_file_path = os.path.join(output_dir, f"{dataset_name}.npy")
    #     np.save(output_file_path, dataset)
    #     print(f"Saved {dataset_name} to {output_file_path}")

def unpack_multiple_systems(hdf5_file_path, output_dir, indices: List[int]):
    """Unpack datasets from an HDF5 file for multiple systems and save them as .npy files."""
    print(f"Unpacking {hdf5_file_path} to {output_dir}")

    # Remove the directory and its contents if it already exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    hdf5_file = h5py.File(hdf5_file_path, "r")
    # Iterate through all groups in the HDF5 file
    for group_idx, group_name in enumerate(hdf5_file.keys()):
        if group_idx not in indices:
            continue

        group = hdf5_file[group_name]
        if not isinstance(group, h5py.Group):
            print(f"Expected group, but got {type(group)} for {group_name}")
            continue
        
        group_dict = {key: np.array(value) for key, value in group.items() if isinstance(value, h5py.Dataset)}
        for key in ORCA_CONVERSION_DICTIONARY.keys():
            if not key in group_dict.keys():
                raise ValueError(f"Expected {key} in group {group_name}, but got {group_dict.keys()}")
        for key, value in DELTA_KEYS.items():
            group_dict[key] = group_dict[value[0]] - group_dict[value[1]]
        group_dict = {ORCA_CONVERSION_DICTIONARY.get(key, key): value for key, value in group_dict.items()}
        np.save(os.path.join(output_dir, f"{group_name}_batch_{group_idx}.npy"), group_dict, allow_pickle=True)
            
    hdf5_file.close()

def main():
    """Main function to parse arguments and unpack HDF5 file."""
    args = parse_arguments()
    # View the structure of the HDF5 file
    if args.view:
        view_hdf5_file(args.hdf5_file_path)

    # Unpack datasets from the HDF5 file and save them as .npy files
    if args.output_dir and args.single_system:
        unpack_single_system(args.hdf5_file_path, args.output_dir, args.indices[0])

    if args.output_dir and not args.single_system:
        unpack_multiple_systems(args.hdf5_file_path, args.output_dir, args.indices)

if __name__ == "__main__":
    main()