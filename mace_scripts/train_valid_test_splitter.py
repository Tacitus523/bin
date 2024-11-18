#!/usr/bin/env python3
# Load the complete dataset from a .extxyz file, split it into training, validation and test set
# and save it in smaller .extxyz files
import argparse
import os
import numpy as np
from ase.io import read, write
from sklearn.model_selection import train_test_split

DATA_FOLDER: str = "/lustre/work/ws/ws1/ka_he8978-dipeptide/training_data/B3LYP_aug-cc-pVTZ_vacuum"
#DATA_FOLDER: str = "/lustre/work/ws/ws1/ka_he8978-thiol_disulfide/training_data/B3LYP_aug-cc-pVTZ_vacuum"
GEOM_FILE: str = "geoms.extxyz" # Name of the file containing the dataset, format: .extxyz
TRAIN_FILE: str = "train.extxyz" # Hardcoded like this in training script train_mace.sh
VALID_FILE: str = "valid.extxyz" # Hardcoded like this in training script train_mace.sh
TEST_FILE: str = "test.extxyz" # Hardcoded like this in training script train_mace.sh

def parse_args():
    parser = argparse.ArgumentParser(description="Split a dataset into training, validation and test set")
    parser.add_argument("-d", "--data-folder", type=str, default=DATA_FOLDER, required=False, help="Path to the folder containing the dataset")
    parser.add_argument("-g", "--geom-file", type=str, default=GEOM_FILE, required=False, help="Name of the file containing the dataset, format: .extxyz")
    parser.add_argument("-t", "--train-file", type=str, default=TRAIN_FILE, required=False, help="Name of the file training set is saved to")
    parser.add_argument("-v", "--valid-file", type=str, default=VALID_FILE, required=False, help="Name of the file validation set is saved to")
    parser.add_argument("-e", "--test-file", type=str, default=TEST_FILE, required=False, help="Name of the file test set is saved to")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    geom_file = os.path.join(args.data_folder, args.geom_file)
    train_file = os.path.join(args.data_folder, args.train_file)
    valid_file = os.path.join(args.data_folder, args.valid_file)
    test_file = os.path.join(args.data_folder, args.test_file)

    # Load the complete dataset from a .extxyz file
    data = read(geom_file, ":", format="extxyz")
    print(f"Loaded {len(data)} configurations from 'geoms.extxyz'")

    # Split the dataset into training, validation and test set
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, shuffle=False)
    train_data, valid_data = train_test_split(train_data, test_size=0.2, random_state=42, shuffle=False)

    # Save the training, validation and test set in smaller .extxyz files
    write(train_file, train_data, format="extxyz")
    write(valid_file, valid_data, format="extxyz")
    write(test_file, test_data, format="extxyz")

if __name__ == "__main__":
    main()