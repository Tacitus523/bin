#!/usr/bin/env python
import argparse
import os

from ase.io import read
from ase import Atoms
import numpy as np
from schnetpack.data import ASEAtomsData

# NAME DB_FILE // NAME MODEL_DIR // NAME MODEL 
EXTXYZ_FILE = "geoms.extxyz"
DB_FILE = "geoms.db"

# Expected property keys in the .extxyz file
ENERGY_KEY = "ref_energy" # eV
FORCES_KEY = "ref_force" # eV/Ang
CHARGES_KEY = "ref_charge" # e
ESP_KEY = "esp" # eV/e
ESP_GRADIENT_KEY = "esp_gradient" # eV/Ang

distance_unit = "Angstrom",
PROPERTY_UNIT_DICT = {
    "energy": "eV", 
    "forces": "eV/Angstrom",
    "partial_charges": "charge", 
    "esp": "eV/charge",
    "electric_field": "eV/Angstrom/charge",
}

def parse_args() -> argparse.Namespace:
    # Argument Parser for command line arguments
    parser = argparse.ArgumentParser("Convert an extxyz file to an ASE database")
    parser.add_argument("-f", "--file", type=str, required=False, default=EXTXYZ_FILE, help="Path to the extxyz file")
    parser.add_argument("-d", "--database", type=str, required=False, default=DB_FILE, help="Path to the database file")
    args = parser.parse_args()
    return args

def convert_extxyz_to_db(args: argparse.Namespace) -> None:
    # READING AND CREATING DB 

    # Read all frames from the extxyz file
    molecules = read(args.file, index=":")  
    property_list = [] 


    # setting up for creating atomsobject // right format for schnetpack
    for molecule in molecules: 

        energy = molecule.info.get(ENERGY_KEY)
        if energy is None: 
            raise ValueError(f"{molecule} has no energy information with key {ENERGY_KEY}")
        
        forces = molecule.arrays.get(FORCES_KEY) 
        if forces is None: 
            raise ValueError(f"{molecule} has no forces information with key {FORCES_KEY}")

        partial_charges = molecule.arrays.get(CHARGES_KEY, np.zeros(len(molecule)))
        
        esp = molecule.arrays.get(ESP_KEY, np.zeros(len(molecule)))

        esp_gradient = molecule.arrays.get(ESP_GRADIENT_KEY)
        if esp_gradient is None: 
            esp_gradient = np.zeros((len(molecule), 3))
        else: 
            esp_gradient = np.zeros((len(molecule), 3))
        electric_field = -esp_gradient
        molecule.arrays["electric_field"] = electric_field

        properties = {"energy": energy, "forces": forces, "partial_charges": partial_charges, "esp": esp, "electric_field": electric_field} # add more properties if needed
        property_list.append(properties)

    # ASEAtomsData.create cant overwrite db file 
    if os.path.exists(args.database):
        os.remove(args.database)

    # creating the new database 
    new_dataset = ASEAtomsData.create(
        args.database,
        distance_unit = distance_unit,
        property_unit_dict=PROPERTY_UNIT_DICT, 
    )

    # adding the properties 
    new_dataset.add_systems(property_list, molecules)

    print(f"Converted {len(molecules)} frames from {args.file} to {args.database}")

def main() -> None:
    args = parse_args()
    convert_extxyz_to_db(args)

if __name__=="__main__":
    main()
