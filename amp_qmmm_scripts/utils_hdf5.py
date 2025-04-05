#!/usr/bin/env python
import h5py
import numpy as np
import os
import argparse
import shutil
from typing import Dict, List

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Unpack datasets from an HDF5 file and save them as .npy files.")
    parser.add_argument("hdf5_file_path", type=str, help="Path to the HDF5 file.")
    parser.add_argument("-o", "--output_dir", required=False, default=None, type=str, help="Output directory for .npy files.")
    parser.add_argument("--view", action="store_true", help="View the structure of the HDF5 file.")
    args = parser.parse_args()

    if not os.path.exists(args.hdf5_file_path):
        parser.error(f"The file {args.hdf5_file_path} does not exist.")
        exit(1)
    
    if not args.view and not args.output_dir:
        args.view = True
    
    return args

def view_hdf5_file(hdf5_file_path):
    def print_hdf5_structure(name, obj):
        if isinstance(obj, h5py.Group):
            print(f"Group: {name}")
        elif isinstance(obj, h5py.Dataset):
            print(f"  Dataset: {name}, Shape: {obj.shape}, Type: {obj.dtype}")

    with h5py.File(hdf5_file_path, "r") as hdf5_file:
        hdf5_file.visititems(print_hdf5_structure)

def unpack_hdf5_to_npy(hdf5_file_path, output_dir):
    """Unpack datasets from an HDF5 file and save them as .npy files."""
    print(f"Unpacking {hdf5_file_path} to {output_dir}")

    # Remove the directory and its contents if it already exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    datasets: Dict[str, List[np.ndarray]] = {}
    hdf5_file = h5py.File(hdf5_file_path, "r")
    # Iterate through all groups in the HDF5 file
    for group_name in hdf5_file.keys():
        group = hdf5_file[group_name]
        if not isinstance(group, h5py.Group):
            print(f"Expected group, but got {type(group)} for {group_name}")
            continue

        # Unpack datasets in the group
        for dataset_name in group.keys():
            dataset = group[dataset_name]
            if not isinstance(dataset, h5py.Dataset):
                print(f"Expected dataset, but got {type(dataset)} for {dataset_name}")
                continue

            if dataset_name not in datasets:
                print(f"  Dataset: {dataset_name}, Shape: {dataset.shape}, Type: {dataset.dtype}")
                datasets[dataset_name] = []

            np.save(os.path.join(output_dir, f"{dataset_name}.npy"), dataset)
            datasets[dataset_name].append(dataset)
            
        break
    hdf5_file.close()
    
    #INFO: Doesn't work, because the amount MM particles differs 
    # datasets = {dataset_name: np.concatenate(dataset) for dataset_name, dataset in datasets.items()}
    # # Save each dataset as a .npy file
    # for dataset_name, dataset in datasets.items():
    #     output_file_path = os.path.join(output_dir, f"{dataset_name}.npy")
    #     np.save(output_file_path, dataset)
    #     print(f"Saved {dataset_name} to {output_file_path}")
        
def main():
    """Main function to parse arguments and unpack HDF5 file."""
    args = parse_arguments()
    # View the structure of the HDF5 file
    if args.view:
        view_hdf5_file(args.hdf5_file_path)

    # Unpack datasets from the HDF5 file and save them as .npy files
    if args.output_dir:
        unpack_hdf5_to_npy(args.hdf5_file_path, args.output_dir)

if __name__ == "__main__":
    main()