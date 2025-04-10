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

# Expected like this in train_amp.py
TRAINING_DIRECTORY = "training"
VALIDATION_DIRECTORY = "validation"
TEST_DIRECTORY = "test"

SYSTEM_NAME = "dalanine"

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Unpack datasets from an HDF5 file and save them as .npy files.")
    parser.add_argument("hdf5_file_path", type=str, help="Path to the HDF5 file.")
    parser.add_argument("-o", "--output_dir", required=False, default=None, type=str, help="Output directory for .npy files.")
    parser.add_argument("-n", "--name", type=str, default=SYSTEM_NAME, help="Name of the system. Default is %s." % SYSTEM_NAME)
    parser.add_argument("-i", "--indices", nargs="+", type=int, default=None, help="Indices of the datasets to unpack.")
    parser.add_argument("-s", "--single_system", action="store_true", help="Unpack only the one dataset.")
    parser.add_argument("--splits", nargs=3, type=float, default=None, help="Split each system into training, validation, and test sets. Provide the split ratios as three floats.")
    parser.add_argument("-v", "--view", action="store_true", help="View the structure of the HDF5 file.")
    args = parser.parse_args()

    if not os.path.exists(args.hdf5_file_path):
        parser.error(f"The file {args.hdf5_file_path} does not exist.")

    if args.output_dir is not None:
        args.output_dir = os.path.abspath(args.output_dir)
    
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

    # Print the parsed arguments
    print("Parsed arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    return args

def view_hdf5_file(hdf5_file_path):
    def print_hdf5_structure(name, obj):
        if isinstance(obj, h5py.Group):
            print(f"Group: {name}")
        elif isinstance(obj, h5py.Dataset):
            print(f"  Dataset: {name}, Shape: {obj.shape}, Type: {obj.dtype}")

    with h5py.File(hdf5_file_path, "r") as hdf5_file:
        hdf5_file.visititems(print_hdf5_structure)

def unpack_single_system(args):
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

def unpack_multiple_systems(args):
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
        for key in ORCA_CONVERSION_DICTIONARY.keys():
            if not key in group_dict.keys():
                raise ValueError(f"Expected {key} in group {group_name}, but got {group_dict.keys()}")
        for key, value in DELTA_KEYS.items():
            group_dict[key] = group_dict[value[0]] - group_dict[value[1]]
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
    # View the structure of the HDF5 file
    if args.view:
        view_hdf5_file(args.hdf5_file_path)

    # Unpack datasets from the HDF5 file and save them as .npy files
    if args.output_dir and args.single_system:
        unpack_single_system(args)

    if args.output_dir and not args.single_system:
        unpack_multiple_systems(args)

if __name__ == "__main__":
    main()