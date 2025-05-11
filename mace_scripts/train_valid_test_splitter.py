#!/usr/bin/env python3
# Load the complete dataset from a .extxyz file, split it into training, validation and test set
# and save it in smaller .extxyz files
import argparse
import os
import numpy as np
from ase.io import read, write
import shutil
from sklearn.model_selection import train_test_split


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

def parse_args():
    parser = argparse.ArgumentParser(description="Split a dataset into training, validation and test set")
    parser.add_argument("-n", "--nsplits", type=int, default=1, required=False, help="Number of splits to perform")
    parser.add_argument("-d", "--data-folder", type=str, default=DATA_FOLDER, required=False, help="Path to the folder containing the dataset")
    parser.add_argument("-g", "--geom-file", dest="geom_file", type=str, default=GEOM_FILE, required=False, help="Name of the file containing the dataset, format: .extxyz")
    parser.add_argument("-s", "--sources-file", dest="sources_file", type=str, default=DATA_SOURCES_FILE, required=False, help="Name of the file containing the data source of each entry")
    parser.add_argument("-t", "--train-file", dest="train_file", type=str, default=TRAIN_FILE, required=False, help="Name of the file training set is saved to")
    parser.add_argument("-v", "--valid-file", dest="valid_file", type=str, default=VALID_FILE, required=False, help="Name of the file validation set is saved to")
    parser.add_argument("-e", "--test-file", dest="test_file", type=str, default=TEST_FILE, required=False, help="Name of the file test set is saved to")
    parser.add_argument("--p_test", type=float, default=0.2, required=False, help="Proportion of the whole dataset used for the test set")
    parser.add_argument("--p_valid", type=float, default=0.2, required=False, help="Proportion of the not-test part of the dataset used for the validation set")
    args = parser.parse_args()
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    return args

def validate_args(args):
    args.data_folder = os.path.abspath(args.data_folder)
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
    return args

def do_train_val_test_split(args):
    # Load the complete dataset from a .extxyz file
    data = read(args.geom_file, ":", format="extxyz")
    print(f"Loaded {len(data)} configurations from {args.geom_file}")

    # Split the dataset into training, validation and test set
    random_state = 42 # Important for equal split of data and data sources
    train_data, test_data = train_test_split(data, test_size=args.p_test, random_state=random_state, shuffle=True)
    train_data, valid_data = train_test_split(train_data, test_size=args.p_valid, random_state=random_state+1, shuffle=True)

    # Save the training, validation and test set in smaller .extxyz files
    write(args.train_file, train_data, format="extxyz")
    write(args.valid_file, valid_data, format="extxyz")
    write( args.test_file, test_data, format="extxyz")

    if args.sources_file is None:
        return
    
    # Split the sources file into training, validation and test set
    with open(args.sources_file, "r") as f:
        data_sources: np.ndarray = np.array([line.strip() for line in f.readlines()], dtype=str)
    
    assert len(data_sources) == len(data), f"Number of lines in sources file {args.sources_file} does not match number of configurations in {args.geom_file}"
    train_sources, test_sources = train_test_split(data_sources, test_size=args.p_test, random_state=random_state, shuffle=True)
    train_sources, valid_sources = train_test_split(train_sources, test_size=args.p_valid, random_state=random_state+1, shuffle=True)
    assert len(train_sources) == len(train_data), f"Number of lines in sources file {args.sources_file} does not match number of configurations in {args.train_file}: {len(train_sources)} != {len(train_data)}"
    assert len(valid_sources) == len(valid_data), f"Number of lines in sources file {args.sources_file} does not match number of configurations in {args.valid_file}: {len(valid_sources)} != {len(valid_data)}"
    assert len(test_sources) == len(test_data), f"Number of lines in sources file {args.sources_file} does not match number of configurations in {args.test_file}: {len(test_sources)} != {len(test_data)}"
    np.savetxt(args.train_sources_file, train_sources, fmt="%s")
    np.savetxt(args.valid_sources_file, valid_sources, fmt="%s")
    np.savetxt(args.test_sources_file, test_sources, fmt="%s")

def split_split(split_file: str, split_sources_file: str, args):
    total_split_data = read(split_file, ":", format="extxyz")

    total_size = len(total_split_data)
    indices = np.arange(total_size)
    np.random.shuffle(indices)
    for split_idx in range(args.nsplits):
        split_indices = indices[split_idx::args.nsplits] # Split the indices into nsplits parts
        split_indices = np.sort(split_indices)
        split_data = [total_split_data[j] for j in split_indices]
        
        os.makedirs(f"split_{split_idx}", exist_ok=True)
        save_file = os.path.join(f"split_{split_idx}", split_file) # Save the split in a new folder
        write(save_file, split_data, format="extxyz")

        if args.sources_file is not None:
            with open(split_sources_file, "r") as f:
                split_sources: np.ndarray = np.array([line.strip() for line in f.readlines()], dtype=str)
            assert len(split_sources) == total_size, f"Number of lines in sources file {split_sources_file} does not match number of configurations in {split_file}: {len(split_sources)} != {total_size}"
            split_sources = split_sources[split_indices]
            save_sources_file = os.path.join(f"split_{split_idx}", split_sources_file)
            np.savetxt(save_sources_file, split_sources, fmt="%s")

def main():
    args = parse_args()
    args = validate_args(args)
    do_train_val_test_split(args)
    if args.nsplits > 1:
        split_split(args.train_file, args.train_sources_file, args)
        split_split(args.valid_file, args.valid_sources_file, args)
        for split_idx in range(args.nsplits):
            shutil.copy(args.test_file, f"split_{split_idx}")
            if args.sources_file is not None:
                shutil.copy(args.test_sources_file, f"split_{split_idx}")

if __name__ == "__main__":
    main()