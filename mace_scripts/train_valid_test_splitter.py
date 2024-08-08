#!/usr/bin/env python3
# Load the complete dataset from a .extxyz file, split it into training, validation and test set
# and save it in smaller .extxyz files
import os
import numpy as np
from ase.io import read, write
from sklearn.model_selection import train_test_split

DATA_FOLDER: str = "/lustre/work/ws/ws1/ka_he8978-dipeptide/training_data/B3LYP_aug-cc-pVTZ_vacuum"
#DATA_FOLDER: str = "/lustre/work/ws/ws1/ka_he8978-thiol_disulfide/training_data/B3LYP_aug-cc-pVTZ_vacuum"
GEOM_FILE: str = "geoms.extxyz"
TRAIN_FILE: str = "train.extxyz"
VALID_FILE: str = "valid.extxyz"
TEST_FILE: str = "test.extxyz"

def main():
    geom_file = os.path.join(DATA_FOLDER, GEOM_FILE)
    train_file = os.path.join(DATA_FOLDER, TRAIN_FILE)
    valid_file = os.path.join(DATA_FOLDER, VALID_FILE)
    test_file = os.path.join(DATA_FOLDER, TEST_FILE)

    # Load the complete dataset from a .extxyz file
    data = read(geom_file, ":", format="extxyz")
    print(f"Loaded {len(data)} configurations from 'geoms.extxyz'")

    # Split the dataset into training, validation and test set
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, valid_data = train_test_split(train_data, test_size=0.2, random_state=42)

    # Save the training, validation and test set in smaller .extxyz files
    write(train_file, train_data, format="extxyz")
    write(valid_file, valid_data, format="extxyz")
    write(test_file, test_data, format="extxyz")

if __name__ == "__main__":
    main()