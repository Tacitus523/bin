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
TRAIN_FILE: str = TRAIN_FILE_PREFIX + SUFFIX # Hardcoded like this in training script train_mace.sh
VALID_FILE: str = VALID_FILE_PREFIX + SUFFIX # Hardcoded like this in training script train_mace.sh
TEST_FILE: str = TEST_FILE_PREFIX + SUFFIX # Hardcoded like this in training script train_mace.sh

def parse_args():
    parser = argparse.ArgumentParser(description="Split a dataset into training, validation and test set")
    parser.add_argument("-n", "--nsplits", type=int, default=1, required=False, help="Number of splits to perform")
    parser.add_argument("-d", "--data-folder", type=str, default=DATA_FOLDER, required=False, help="Path to the folder containing the dataset")
    parser.add_argument("-g", "--geom-file", dest="geom_file", type=str, default=GEOM_FILE, required=False, help="Name of the file containing the dataset, format: .extxyz")
    parser.add_argument("-t", "--train-file", dest="train_file", type=str, default=TRAIN_FILE, required=False, help="Name of the file training set is saved to")
    parser.add_argument("-v", "--valid-file", dest="valid_file", type=str, default=VALID_FILE, required=False, help="Name of the file validation set is saved to")
    parser.add_argument("-e", "--test-file", dest="test_file", type=str, default=TEST_FILE, required=False, help="Name of the file test set is saved to")
    parser.add_argument("--p_test", type=float, default=0.2, required=False, help="Proportion of the whole dataset used for the test set")
    parser.add_argument("--p_valid", type=float, default=0.2, required=False, help="Proportion of the not-test part of the dataset used for the validation set")
    args = parser.parse_args()
    return args

def do_train_val_test_split(args):
    geom_file = os.path.join(args.data_folder, args.geom_file)
    # Load the complete dataset from a .extxyz file
    data = read(geom_file, ":", format="extxyz")
    print(f"Loaded {len(data)} configurations from 'geoms.extxyz'")

    # Split the dataset into training, validation and test set
    train_data, test_data = train_test_split(data, test_size=args.p_test, random_state=42, shuffle=False)
    train_data, valid_data = train_test_split(train_data, test_size=args.p_valid, random_state=42, shuffle=False)

    # Save the training, validation and test set in smaller .extxyz files
    valid_file = os.path.join(os.getcwd(), args.valid_file)
    train_file = os.path.join(os.getcwd(), args.train_file)
    test_file = os.path.join(os.getcwd(), args.test_file)
    write(train_file, train_data, format="extxyz")
    write(valid_file, valid_data, format="extxyz")
    write(test_file, test_data, format="extxyz")

def split_split(split_file: str, args: argparse.Namespace):
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

def main():
    args = parse_args()
    do_train_val_test_split(args)
    if args.nsplits > 1:
        split_split(args.train_file, args)
        split_split(args.valid_file, args)
        for split_idx in range(args.nsplits):
            shutil.copy(args.test_file, f"split_{split_idx}")

if __name__ == "__main__":
    main()