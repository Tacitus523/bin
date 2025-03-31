#!/usr/bin/env python
import argparse
from ase import Atoms
from ase.io import read
from ase.db import connect
import datetime
import os 
import numpy as np
from typing import List, Dict

#TODO: Dipole moments properly

# Default file names
EXTXYZ_FILE = "geoms.extxyz"
DB_FILE = "geoms.db"

# Default reference method
REFERENCE_METHOD = "DFTB/ob2-1-1"

# Expected property keys in the .extxyz file
N_STATES_KEY = "n_states"
ENERGY_KEY_PREFIX = "ref_energy" # ref_energy_0, ref_energy_1, ...
FORCES_KEY = "ref_force"
ESP_KEY = "esp"
ELECTRIC_FIELD_KEY = "e_field"
ELECTRIC_FIELD_GRADIENT_KEY = "e_field_grad"
OSCILLATOR_STRENGTH_KEY_PREFIX = "ref_osc_str" # ref_osc_str_0, ref_osc_str_1, ...

eV_to_Ha = 1/27.21138602 
Ang_to_Bohr = 1/0.52917721067


def parse_args() -> argparse.Namespace:
    # Argument Parser for command line arguments
    parser = argparse.ArgumentParser("Convert an extxyz file to an ASE database", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--file", type=str, required=False, default=EXTXYZ_FILE, help="Path to the extxyz file, default: %(default)s")
    parser.add_argument("-d", "--database", type=str, required=False, default=DB_FILE, help="Path to the database file, default: %(default)s")
    args = parser.parse_args()
    return args

def convert_extxyz_to_db(args: argparse.Namespace) -> None:
    # READING AND CREATING DB 

    # Read all frames from the extxyz file
    molecule_list: List[Atoms] = read(args.file, index=":")  
    property_list: List[Dict[str, np.ndarray]] = [] 


    # setting up for creating atomsobject // right format for schnetpack
    for molecule_idx, molecule in enumerate(molecule_list): 
        # Reading in number of states
        n_states = molecule.info.get(N_STATES_KEY)
        if n_states is None: 
            raise ValueError(f"Could not find n_states key {N_STATES_KEY} in the .extxyz file for molecule {molecule_idx}")
        else: 
            n_states = int(n_states)

        # Reading in energies
        energies = []
        for state_idx in range(n_states): 
            energy_key = f"{ENERGY_KEY_PREFIX}_{state_idx}"
            energy = molecule.info.get(energy_key)
            if energy is None: 
                raise ValueError(f"Could not find energy key {energy_key} in the .extxyz file for molecule {molecule_idx}")
            else: 
                energies.append(energy)
        energies = np.array(energies) # Hartree, shape (n_states,)
        
        # Reading in forces 
        forces = molecule.arrays.get(FORCES_KEY)
        if forces is None: 
            raise ValueError(f"Could not find forces key {FORCES_KEY} in the .extxyz file for molecule {molecule_idx}")
        else: 
            forces = np.array(forces)
            forces = np.stack([forces[:, i:i+3] for i in range(0, n_states*3, 3)], axis=0) # Hartree/Bohr, shape (n_states, n_atoms, 3)
            gradients = forces * -1 # Hartree/Bohr, shape (n_states, n_atoms, 3)
            has_gradients = np.ones(((1,)), dtype=np.int64)
        
        # electrostatic potential
        esp = molecule.arrays.get(ESP_KEY)
        if esp is None:
            raise ValueError(f"Could not find esp key {ESP_KEY} in the .extxyz file for molecule {molecule_idx}")
        else: 
            esp = np.array(esp) # Hartree/e, shape (n_atoms,)

        # electric_field
        electric_field = molecule.arrays.get(ELECTRIC_FIELD_KEY)
        if electric_field is None: 
            raise ValueError(f"Could not find electric field key {ELECTRIC_FIELD_KEY} in the .extxyz file for molecule {molecule_idx}")
        else: 
            electric_field = np.array(electric_field) # Hartree/Bohr/e, shape (n_atoms, 3)

        # electric_field_gradient
        electric_field_gradient = molecule.arrays.get(ELECTRIC_FIELD_GRADIENT_KEY)
        if electric_field_gradient is None:
            raise ValueError(f"Could not find electric field gradient key {ELECTRIC_FIELD_GRADIENT_KEY} in the .extxyz file for molecule {molecule_idx}")
        else:
            electric_field_gradient = np.array(electric_field_gradient)
            electric_field_gradient = np.stack([electric_field_gradient[:, i:i+3] for i in range(0, 9, 3)], axis=1) # Hartree/Bohr^2/e, shape (n_atoms, 3, 3)

        # oscillator strength
        osc_strengths = []
        for state_idx in range(n_states):
            osc_str_key = f"{OSCILLATOR_STRENGTH_KEY_PREFIX}_{state_idx}"
            osc_str = molecule.info.get(osc_str_key)
            if osc_str is None:
                raise ValueError(f"Could not find oscillator strength key {osc_str_key} in the .extxyz file for molecule {molecule_idx}")
            else:
                osc_strengths.append(osc_str)
        osc_strengths = np.array(osc_strengths) # unitless, shape (n_states,)

        # dummy dipole moments
        n_dipole_moments = n_states * (n_states - 1) // 2
        dipole_moments = np.zeros((n_states, 3))

        properties = {
            "energy": energies,
            "forces": forces,
            "gradients": gradients,
            "has_gradients": has_gradients,
            "esp": esp,
            "electric_field": electric_field,
            "dFdr": electric_field_gradient,
            "oscillator_strength": osc_strengths,
            "dipoles": dipole_moments
        }
        property_list.append(properties)


    if os.path.exists(args.database):
        os.remove(args.database)

    metadata = {
        "n_singlets": n_states,
        "n_triplets": 0,
        "reference method": REFERENCE_METHOD,
        "coordinates_units": "Angstrom",
        "energy_units": "Ha",
        "force_units": "Ha/Bohr",
        "esp_units": "Ha/e",
        "electric_field_units": "Ha/Bohr/e",
        "electric_field_gradient_units": "Ha/Bohr^2/e",
        "oscillator_strength_units": "unitless",
        "dipole_units": "dummy"
    }

    # WRITING TO DB
    with connect(args.database) as db: # with context manager to enforce single transaction
        db.metadata = metadata
        for molecule, properties in zip(molecule_list, property_list):
            db.write(molecule, data=properties)

def main() -> None:
    args = parse_args()
    convert_extxyz_to_db(args)

if __name__=="__main__":
    main()
