#!/usr/bin/env python3
# Load the complete dataset from a .extxyz file, split it into training, validation and test set
# and save it in smaller .extxyz files
import argparse
import os
import numpy as np
import shutil
import sys
from typing import List, Optional, Tuple

from ase import Atoms
from ase.io import read, write
from sklearn.model_selection import train_test_split, KFold


TRAIN_FILE_PREFIX: str = "train" # Prefix of the training set files
VALID_FILE_PREFIX: str = "valid" # Prefix of the validation set files
TEST_FILE_PREFIX: str = "test" # Prefix of the test set files
SUFFIX: str = ".extxyz" # Extended XYZ file format

#DATA_FOLDER: str = "/lustre/work/ws/ws1/ka_he8978-dipeptide/training_data/B3LYP_aug-cc-pVTZ_vacuum"
#DATA_FOLDER: str = "/lustre/work/ws/ws1/ka_he8978-thiol_disulfide/training_data/B3LYP_aug-cc-pVTZ_vacuum"
DATA_FOLDER: str = os.getcwd()
GEOM_FILE: str = "geoms.extxyz" # Name of the file containing the dataset, format: .extxyz
DATA_SOURCES_FILE: str = "data_sources.txt" # Name of the file containing the data source of each entry
TRAIN_FILE: str = TRAIN_FILE_PREFIX + SUFFIX # Hardcoded like this in training script train_mace.sh
VALID_FILE: str = VALID_FILE_PREFIX + SUFFIX # Hardcoded like this in training script train_mace.sh
TEST_FILE: str = TEST_FILE_PREFIX + SUFFIX # Hardcoded like this in training script train_mace.sh
TRAIN_SOURCES_FILE: str = "{data_sources_prefix}_{train_prefix}{data_sources_suffix}"
VALID_SOURCES_FILE: str = "{data_sources_prefix}_{valid_prefix}{data_sources_suffix}"
TEST_SOURCES_FILE: str = "{data_sources_prefix}_{test_prefix}{data_sources_suffix}"
TRAIN_INDICES_FILE: str = "{train_prefix}_indices.txt"
VALID_INDICES_FILE: str = "{valid_prefix}_indices.txt"
TEST_INDICES_FILE: str = "{test_prefix}_indices.txt"

def parse_args():
    parser = argparse.ArgumentParser(description="Split a dataset into training, validation and test set or subsample training data")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Subparser for splitting dataset
    split_parser = subparsers.add_parser("split", help="Split a dataset into training, validation and test set")
    split_parser.add_argument("-n", "--nsplits", type=int, default=1, required=False, help="Number of splits to perform")
    split_parser.add_argument("-d", "--data-folder", type=str, default=DATA_FOLDER, required=False, help="Path to the folder containing the dataset")
    split_parser.add_argument("-g", "--geom-file", dest="geom_file", type=str, default=GEOM_FILE, required=False, help="Name of the file containing the dataset, format: .extxyz")
    split_parser.add_argument("-s", "--sources-file", dest="sources_file", type=str, default=DATA_SOURCES_FILE, required=False, help="Name of the file containing the data source of each entry")
    split_parser.add_argument("-t", "--train-file", dest="train_file", type=str, default=TRAIN_FILE, required=False, help="Name of the file training set is saved to")
    split_parser.add_argument("-v", "--valid-file", dest="valid_file", type=str, default=VALID_FILE, required=False, help="Name of the file validation set is saved to")
    split_parser.add_argument("-e", "--test-file", dest="test_file", type=str, default=TEST_FILE, required=False, help="Name of the file test set is saved to")
    split_parser.add_argument("--p_test", type=float, default=0.2, required=False, help="Proportion of the whole dataset used for the test set")
    split_parser.add_argument("--p_valid", type=float, default=0.2, required=False, help="Proportion of the not-test part of the dataset used for the validation set")
    split_parser.add_argument("--kfold", action="store_true", default=False, required=False, 
        help="Only with --nsplits: Perform K-Fold cross-validation split with overlapping training sets. Default: Overlap validation sets instead of training sets.")
    split_parser.add_argument("--random-seed", type=int, default=42, required=False, help="Random seed for reproducible splitting")
    
    # Subparser for subsampling training data
    subsample_parser = subparsers.add_parser("subsample", help="Subsample training data from an already split dataset")
    subsample_parser.add_argument("-n", "--num-samples", type=int, required=True, help="Number of samples to extract from the training set")
    subsample_parser.add_argument("-d", "--data-folder", type=str, default=DATA_FOLDER, required=False, help="Path to the folder containing the split dataset")
    subsample_parser.add_argument("-t", "--train-file", dest="train_file", type=str, default=TRAIN_FILE, required=False, help="Name of the training file to subsample")
    subsample_parser.add_argument("-o", "--output-file", dest="output_file", type=str, required=False, help="Name of the output file (default: train_subsampled_N.extxyz)")
    subsample_parser.add_argument("-s", "--sources-file", dest="sources_file", type=str, required=False, help="Name of the training sources file to subsample")
    subsample_parser.add_argument("--random-seed", type=int, default=42, required=False, help="Random seed for reproducible subsampling")
    
    args = parser.parse_args()
    return args

def validate_args(args):
    if args.command is None:
        print("No command specified. Use 'split' or 'subsample'.")
        sys.exit(1)
    
    args.data_folder = os.path.abspath(args.data_folder)
    
    if args.command == 'split':
        args.geom_file = os.path.abspath(os.path.join(args.data_folder, args.geom_file))
        args.sources_file = os.path.abspath(os.path.join(args.data_folder, args.sources_file))
        data_sources_prefix, data_sources_suffix = os.path.splitext(args.sources_file)
        data_sources_prefix = os.path.basename(data_sources_prefix) # Remove the path from the sources file
        args.train_sources_file = TRAIN_SOURCES_FILE.format(
            data_sources_prefix=data_sources_prefix, 
            train_prefix=TRAIN_FILE_PREFIX, 
            data_sources_suffix=data_sources_suffix)
        args.valid_sources_file = VALID_SOURCES_FILE.format(
            data_sources_prefix=data_sources_prefix, 
            valid_prefix=VALID_FILE_PREFIX, 
            data_sources_suffix=data_sources_suffix)
        args.test_sources_file = TEST_SOURCES_FILE.format(
            data_sources_prefix=data_sources_prefix, 
            test_prefix=TEST_FILE_PREFIX, 
            data_sources_suffix=data_sources_suffix)

        args.train_indices_file = TRAIN_INDICES_FILE.format(
            train_prefix=TRAIN_FILE_PREFIX)
        args.valid_indices_file = VALID_INDICES_FILE.format(
            valid_prefix=VALID_FILE_PREFIX)
        args.test_indices_file = TEST_INDICES_FILE.format(
            test_prefix=TEST_FILE_PREFIX)

        if not os.path.exists(args.data_folder):
            raise FileNotFoundError(f"Data folder {args.data_folder} does not exist")
        if not os.path.exists(os.path.join(args.data_folder, args.geom_file)):
            raise FileNotFoundError(f"Geometry file {args.geom_file} does not exist in {args.data_folder}")
        if not os.path.exists(os.path.join(args.data_folder, args.sources_file)):
            print(f"WARNING: Sources file {args.sources_file} does not exist in {args.data_folder}. Not splitting the sources file.")
            args.sources_file = None
            args.train_sources_file = None
            args.valid_sources_file = None
            args.test_sources_file = None

        if args.p_test < 0 or args.p_test > 1:
            raise ValueError("Proportion of the test set must be between 0 and 1")
        if args.p_valid < 0 or args.p_valid > 1:
            raise ValueError("Proportion of the validation set must be between 0 and 1")
        if args.p_test + args.p_valid > 1:
            raise ValueError("Proportion of the test and validation set must be less than 1")
        if args.nsplits < 1:
            raise ValueError("Number of splits must be at least 1")
            
    elif args.command == 'subsample':
        args.train_file = os.path.abspath(os.path.join(args.data_folder, args.train_file))
        
        if not os.path.exists(args.data_folder):
            raise FileNotFoundError(f"Data folder {args.data_folder} does not exist")
        if not os.path.exists(args.train_file):
            raise FileNotFoundError(f"Training file {args.train_file} does not exist")
            
        if args.sources_file is not None:
            args.sources_file = os.path.abspath(os.path.join(args.data_folder, args.sources_file))
            if not os.path.exists(args.sources_file):
                print(f"WARNING: Sources file {args.sources_file} does not exist. Proceeding without sources file.")
                args.sources_file = None
                
        if args.num_samples < 1:
            raise ValueError("Number of samples must be at least 1")
            
    return args

def read_data(args: argparse.Namespace) -> Tuple[List[Atoms], Optional[np.ndarray]]:
    # Read the dataset from the .extxyz file
    data: List[Atoms] = read(args.geom_file, ":", format="extxyz")

    # Check if the sources file exists and read it
    if hasattr(args, 'sources_file') and args.sources_file is not None:
        data_sources: np.ndarray = np.loadtxt(args.sources_file, dtype=str, delimiter="*") # Dummy delimiter, should not occur
        assert len(data_sources) == len(data), \
            f"Number of lines in sources file {args.sources_file} does not match number of configurations in {args.geom_file}: {len(data_sources)} != {len(data)}"
    else:
        data_sources = None
    print(f"Loaded {len(data)} configurations from {args.geom_file}")
    return data, data_sources

def do_train_val_test_split(data: List[Atoms], args: argparse.Namespace, data_sources: Optional[np.ndarray] = None):
    # Split the dataset into training, validation and test set
    random_state = args.random_seed # Important for equal split of data and data sources
    data_indices = np.arange(len(data)) # For saving the indices of the configurations in the .extxyz files
    train_indices, test_indices = train_test_split(data_indices, test_size=args.p_test, random_state=random_state, shuffle=True)
    train_indices, valid_indices = train_test_split(train_indices, test_size=args.p_valid, random_state=random_state+1, shuffle=True)
    train_data = [data[i] for i in train_indices]
    valid_data = [data[i] for i in valid_indices]
    test_data = [data[i] for i in test_indices]

    # Save the training, validation and test set in smaller .extxyz files
    write(args.train_file, train_data, format="extxyz")
    write(args.valid_file, valid_data, format="extxyz")
    write(args.test_file, test_data, format="extxyz")

    np.savetxt(args.train_indices_file, train_indices, fmt="%d")
    np.savetxt(args.valid_indices_file, valid_indices, fmt="%d")
    np.savetxt(args.test_indices_file, test_indices, fmt="%d")

    if data_sources is None:
        return

    train_sources = data_sources[train_indices]
    valid_sources = data_sources[valid_indices]
    test_sources = data_sources[test_indices]
    assert len(train_sources) == len(train_data), f"Number of lines in sources file {args.sources_file} does not match number of configurations in {args.train_file}: {len(train_sources)} != {len(train_data)}"
    assert len(valid_sources) == len(valid_data), f"Number of lines in sources file {args.sources_file} does not match number of configurations in {args.valid_file}: {len(valid_sources)} != {len(valid_data)}"
    assert len(test_sources) == len(test_data), f"Number of lines in sources file {args.sources_file} does not match number of configurations in {args.test_file}: {len(test_sources)} != {len(test_data)}"
    np.savetxt(args.train_sources_file, train_sources, fmt="%s")
    np.savetxt(args.valid_sources_file, valid_sources, fmt="%s")
    np.savetxt(args.test_sources_file, test_sources, fmt="%s")

def split_split(split_file: str, split_sources_file: str, args):
    total_split_data = read(split_file, ":", format="extxyz")

    os.makedirs(f"splits", exist_ok=True)

    total_size = len(total_split_data)
    indices = np.arange(total_size)
    np.random.seed(args.random_seed)
    np.random.shuffle(indices)
    for split_idx in range(args.nsplits):
        split_indices = indices[split_idx::args.nsplits] # Split the indices into nsplits parts
        split_indices = np.sort(split_indices)
        split_data = [total_split_data[j] for j in split_indices]
        
        split_folder = os.path.join("splits", f"split_{split_idx}")
        os.makedirs(split_folder, exist_ok=True)
        save_file = os.path.join(split_folder, split_file) # Save the split in a new folder
        write(save_file, split_data, format="extxyz")

        if args.sources_file is not None:
            with open(split_sources_file, "r") as f:
                split_sources: np.ndarray = np.array([line.strip() for line in f.readlines()], dtype=str)
            assert len(split_sources) == total_size, f"Number of lines in sources file {split_sources_file} does not match number of configurations in {split_file}: {len(split_sources)} != {total_size}"
            split_sources = split_sources[split_indices]
            save_sources_file = os.path.join(split_folder, split_sources_file)
            np.savetxt(save_sources_file, split_sources, fmt="%s")

def do_subsample_training(args: argparse.Namespace):
    """Subsample N entries from an existing training dataset."""
    # Read the training data
    train_data: List[Atoms] = read(args.train_file, ":", format="extxyz")
    print(f"Loaded {len(train_data)} configurations from {args.train_file}")
    
    if args.num_samples > len(train_data):
        raise ValueError(f"Requested {args.num_samples} samples but training file only contains {len(train_data)} entries")
    
    # Set random seed for reproducibility
    np.random.seed(args.random_seed)
    
    # Randomly select indices for subsampling
    all_indices = np.arange(len(train_data))
    selected_indices = np.random.choice(all_indices, size=args.num_samples, replace=False)
    selected_indices = np.sort(selected_indices)
    
    # Extract subsampled data
    subsampled_data = [train_data[i] for i in selected_indices]
    
    # Save subsampled training data
    write(args.output_file, subsampled_data, format="extxyz")
    print(f"Saved {len(subsampled_data)} subsampled configurations to {args.output_file}")
    
    # Save indices for reference
    indices_file = args.output_file.replace(".extxyz", "_indices.txt")
    np.savetxt(indices_file, selected_indices, fmt="%d")
    print(f"Saved selected indices to {indices_file}")
    
    # Handle sources file if it exists
    if args.sources_file is not None:
        with open(args.sources_file, "r") as f:
            train_sources = np.array([line.strip() for line in f.readlines()], dtype=str)
        
        if len(train_sources) != len(train_data):
            print(f"WARNING: Sources file length ({len(train_sources)}) doesn't match training data length ({len(train_data)}). Skipping sources subsampling.")
        else:
            subsampled_sources = train_sources[selected_indices]
            sources_output_file = args.output_file.replace(".extxyz", "_sources.txt")
            np.savetxt(sources_output_file, subsampled_sources, fmt="%s")
            print(f"Saved subsampled sources to {sources_output_file}")

def do_kfold_split(data: List[Atoms], args: argparse.Namespace, data_sources: Optional[np.ndarray] = None):
    # Perform K-Fold cross-validation split
    random_state = args.random_seed # Important for equal split of data and data sources
    data_indices = np.arange(len(data)) # For saving the indices of the configurations in the .extxyz files
    train_indices, test_indices = train_test_split(data_indices, test_size=args.p_test, random_state=random_state, shuffle=True)
    test_data = [data[i] for i in test_indices]
    if data_sources is not None:
        test_sources = data_sources[test_indices]
    

    kf = KFold(n_splits=args.nsplits, shuffle=True, random_state=random_state+1)
    for split_idx, (train_indices, valid_indices) in enumerate(kf.split(train_indices)):
        if not args.kfold:
            # Overlap validation sets instead of training sets when not using K-Fold
            train_indices, valid_indices = valid_indices, train_indices
        train_split_data = [data[i] for i in train_indices]
        valid_split_data = [data[i] for i in valid_indices]

        split_folder = os.path.join("kfold_splits", f"split_{split_idx}")
        os.makedirs(split_folder, exist_ok=True)
        train_file = os.path.join(split_folder, args.train_file)
        valid_file = os.path.join(split_folder, args.valid_file)
        test_file = os.path.join(split_folder, args.test_file)
        write(train_file, train_split_data, format="extxyz")
        write(valid_file, valid_split_data, format="extxyz")
        write(test_file, test_data, format="extxyz")

        train_indices_file = os.path.join(split_folder, args.train_indices_file)
        valid_indices_file = os.path.join(split_folder, args.valid_indices_file)
        test_indices_file = os.path.join(split_folder, args.test_indices_file)
        np.savetxt(train_indices_file, train_indices, fmt="%d")
        np.savetxt(valid_indices_file, valid_indices, fmt="%d")
        np.savetxt(test_indices_file, test_indices, fmt="%d")

        if data_sources is not None:
            train_split_sources = data_sources[train_indices]
            valid_split_sources = data_sources[valid_indices]

            train_sources_file = os.path.join(split_folder, args.train_sources_file)
            valid_sources_file = os.path.join(split_folder, args.valid_sources_file)
            test_sources_file = os.path.join(split_folder, args.test_sources_file)

            np.savetxt(train_sources_file, train_split_sources, fmt="%s")
            np.savetxt(valid_sources_file, valid_split_sources, fmt="%s")
            np.savetxt(test_sources_file, test_sources, fmt="%s")

def main():
    args = parse_args()
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    
    if args.command == "split":
        args = validate_args(args)
        data, data_sources = read_data(args)
        
        if args.nsplits == 1:
            do_train_val_test_split(data, args, data_sources)
        if args.nsplits > 1:
            do_kfold_split(data, args, data_sources)
        # Deprecated: split the train, valid and test set into multiple splits
        # split_split(args.train_file, args.train_sources_file, args)
        # split_split(args.valid_file, args.valid_sources_file, args)
        # for split_idx in range(args.nsplits):
        #     shutil.copy(args.test_file, os.path.join("splits", f"split_{split_idx}"))
        #     if args.sources_file is not None:
        #         shutil.copy(args.test_sources_file, os.path.join("splits", f"split_{split_idx}"))
    
    elif args.command == "subsample":
        args = validate_args(args)
        do_subsample_training(args)


if __name__ == "__main__":
    main()