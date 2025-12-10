#!/usr/bin/env python
import argparse
from ase.io import read
from schnetpack.data import ASEAtomsData
from ase import Atoms
import os 
import numpy as np 
import schnetpack as spk
import schnetpack.transform as trn
import torch
import torchmetrics
import pytorch_lightning as pl
from ase import io
import matplotlib.pyplot as plt 
from schnetpack.representation import FieldSchNet




# NAME DB_FILE // NAME MODEL_DIR // NAME MODEL 
EXTXYZ_FILE = "geoms.extxyz"
DB_FILE = "v_data.db"

# Expected property keys in the .extxyz file
ENERGY_KEY = "ref_energy"
FORCES_KEY = "ref_force"
CHARGES_KEY = "ref_charge"
ESP_KEY = "esp"
ELECTRIC_FIELD_KEY = "electric_field"
eV_to_Ha = 1/27.21138602 
Ang_to_Bohr = 1/0.52917721067


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
    atoms_list = read(args.file, index=":")  
    property_list = [] 


    # setting up for creating atomsobject // right format for schnetpack
    for atoms in atoms_list: 
        positions = atoms.positions * Ang_to_Bohr # Ang --> Bohr
        numbers = atoms.numbers

        # Reading in energy // array for torch 
        energy = atoms.info.get(ENERGY_KEY)
        if energy is None: 
            energy = np.zeros(1)
            print(atoms, "has no energy")
        else:    
            energy = np.array([energy]) * eV_to_Ha # eV --> Ha
        
        # Reading in forces 
        forces = atoms.arrays.get(FORCES_KEY) 
        if forces is None: 
            forces = np.zeros((len(atoms), 3))
            print(atoms, "has no forces")
        else: 
            forces = np.array(forces) * eV_to_Ha / Ang_to_Bohr # eV/Ang --> Ha/Bohr


        # Reading in charges tst 
        partial_charges = atoms.arrays.get(CHARGES_KEY)
        if partial_charges is None: 
            partial_charges =  np.zeros(len(atoms))
            print(atoms, "has no partial_charges")
        else: 
            partial_charges = np.array(partial_charges) #### CONV NECESSARY FOR ASE 
        
        # electrostatic potential
        esp = atoms.arrays.get(ESP_KEY)
        if esp is None: 
            esp = np.zeros(len(atoms))
            print(atoms, "has no esp")
        else: 
            esp = np.array(esp) * eV_to_Ha # eV/e --> eV 

        # esp gradient
        electric_field = atoms.arrays.get(ELECTRIC_FIELD_KEY)
        if electric_field is None: 
            electric_field = np.zeros((len(atoms), 3))
            print(atoms, "has no esp gradient")
        else: 
            electric_field = np.zeros((len(atoms), 3)) # (-1) * np.array(electric_field) * eV_to_Ha/Ang_to_Bohr# Sind schon in eV/Ang? 

        # Creating the atoms object
        ats = Atoms(positions=positions, numbers=numbers)
        properties = {"energy": energy, "forces": forces, "partial_charges":partial_charges, "esp":esp, "electric_field": electric_field} # Weitere Feature dann hier hinzufügen
        property_list.append(properties)

    # print(property_list[0])


    # ASEAtomsData.create cant overwrite db file 
    if os.path.exists(args.database):
        os.remove(args.database)

    # creating the new database 
    new_dataset = ASEAtomsData.create(
        args.database,
        distance_unit = "Bohr",
        property_unit_dict={"energy":"Hartree", "forces":"Hartree/Bohr", "partial_charges":"eV", "esp": "Hartree", "electric_field": "Hartree/Bohr"}, 
    )
    # adding the properties 
    for atoms, properties in zip(atoms_list, property_list):
        new_dataset.add_system(atoms, **properties)

def main() -> None:
    args = parse_args()
    convert_extxyz_to_db(args)

if __name__=="__main__":
    main()
