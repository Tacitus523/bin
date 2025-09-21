#!/usr/bin/env python3
import argparse
import json
import os
import numpy as np
import sys
from typing import Dict, List, Any, Tuple, Optional, Union

from ase import Atoms
from ase.io import read, write
"""
Merge vacuum and environment .extxyz files, prefixing properties and calculating differences.
"""

# Default config values
VACUUM_FILE: str = "vacuum.extxyz"
ENVIRONMENT_FILE: str = "environment.extxyz"
OUTFILE: str = "merged_geoms.extxyz"

# Property keys in .extxyz file that are common for both vacuum and environment
COMMON_KEYS = ["numbers", "positions", "pbc", "cell"]

# Property keys that should be prefixed
PREFIX_VACUUM = "vacuum_"
PREFIX_ENV = "env_"

# Property keys that should have differences calculated (environment - vacuum)
DIFF_KEYS = ["ref_energy", "ref_force", "ref_dipole", "ref_quadrupole", "ref_charge"]
DIFF_PREFIX = "diff_"  # Prefix for difference properties

# ESP related keys that should be excluded from vacuum properties
ESP_KEYS = ["esp", "esp_gradient"]

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Merge vacuum and environment extxyz files")
    ap.add_argument("-c", "--conf", default=None, type=str, dest="config_path", 
                   action="store", required=False, help="Path to config file, default: None", 
                   metavar="config")
    ap.add_argument("-v", "--vacuum", default=VACUUM_FILE, type=str, required=False,
                   help=f"Path to vacuum extxyz file, default: {VACUUM_FILE}", metavar="vacuum")
    ap.add_argument("-e", "--environment", default=ENVIRONMENT_FILE, type=str, required=False,
                   help=f"Path to environment extxyz file, default: {ENVIRONMENT_FILE}", metavar="environment")
    ap.add_argument("-o", "--output", default=OUTFILE, type=str, required=False,
                   help=f"Path to output merged file, default: {OUTFILE}", metavar="output")
    ap.add_argument("--tolerance", default=1e-3, type=float, required=False,
                   help="Tolerance for checking identity of positions, default: 1e-3", metavar="tolerance")
    args = ap.parse_args()
    return args

def read_config(args: argparse.Namespace) -> Dict[str, Any]:
    config_data = {}
    config_path = args.config_path

    if config_path is not None:
        try:
            with open(config_path, 'r') as config_file:
                config_data = json.load(config_file)
        except FileNotFoundError:
            print(f"Config file {config_path} not found.")
            exit(1)

    # Set defaults or use config file values
    config_data.setdefault("VACUUM_FILE", args.vacuum)
    config_data.setdefault("ENVIRONMENT_FILE", args.environment)
    config_data.setdefault("OUTFILE", args.output)
    config_data.setdefault("TOLERANCE", args.tolerance)

    assert config_data["VACUUM_FILE"] != config_data["ENVIRONMENT_FILE"], \
        "Vacuum and environment files must be different."
    assert config_data["VACUUM_FILE"] != config_data["OUTFILE"], \
        "Vacuum file and output file must be different."
    assert config_data["ENVIRONMENT_FILE"] != config_data["OUTFILE"], \
        "Environment file and output file must be different."

    return config_data

def read_extxyz(filename: str) -> List[Atoms]:
    """Read extxyz file and return list of atoms objects."""
    try:
        molecules = read(filename, index=":")
        if isinstance(molecules, Atoms):
            molecules = [molecules]
        elif not isinstance(molecules, list):
            raise ValueError(f"Unexpected type for molecules: {type(molecules)}")
        return molecules
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        sys.exit(1)

def check_structural_identity(vac_molecules: List[Atoms], env_molecules: List[Atoms], 
                             tolerance: float = 1e-6) -> None:
    """
    Check if vacuum and environment molecules have the same structure (numbers and positions).
    """
    if len(vac_molecules) != len(env_molecules):
        raise ValueError(f"Number of structures doesn't match: vacuum ({len(vac_molecules)}) "
                         f"vs environment ({len(env_molecules)})")
    
    for i, (vac_mol, env_mol) in enumerate(zip(vac_molecules, env_molecules)):
        if len(vac_mol) != len(env_mol):
            raise ValueError(f"Number of atoms doesn't match in structure {i}: "
                             f"vacuum ({len(vac_mol)}) vs environment ({len(env_mol)})")
        
        if not np.array_equal(vac_mol.numbers, env_mol.numbers):
            raise ValueError(f"Atomic numbers don't match in structure {i}")
        
        pos_diff = np.abs(vac_mol.positions - env_mol.positions).max()
        if pos_diff > tolerance:
            print(f"Warning: Max position difference in structure {i}: {pos_diff:.8f} (tolerance: {tolerance})")
            print("This may indicate different structures or different coordinate origins.")
            user_input = input("Continue anyway? (y/n): ").strip().lower()
            if user_input != "y":
                sys.exit(1)

def merge_molecules(vac_molecules: List[Atoms], env_molecules: List[Atoms]) -> List[Atoms]:
    """
    Merge vacuum and environment molecules, add prefixes to non-redundant properties,
    and calculate differences between environment and vacuum properties.
    Remove original ref_ properties after they've been prefixed and differences calculated.
    """
    merged_molecules = []
    
    for i, (vac_mol, env_mol) in enumerate(zip(vac_molecules, env_molecules)):
        # Create a new molecule based on environment molecule (keeping common properties)
        merged_mol = env_mol.copy()
        
        # Process info dictionary properties
        for key in vac_mol.info:
            if key in COMMON_KEYS:
                # Verify common keys match
                assert np.array_equal(vac_mol.info[key], env_mol.info[key]), f"Mismatch in {key} for structure {i}"
            elif key in ESP_KEYS:
                # Skip ESP-related properties for vacuum
                continue
            elif key in DIFF_KEYS and key in env_mol.info:
                # Calculate and store differences for specified properties
                diff_value = env_mol.info[key] - vac_mol.info[key]
                merged_mol.info[DIFF_PREFIX + key] = diff_value
                
                # Store values with prefixes
                merged_mol.info[PREFIX_VACUUM + key] = vac_mol.info[key]
                merged_mol.info[PREFIX_ENV + key] = env_mol.info[key]
                
                # Remove original property
                if key in merged_mol.info:
                    del merged_mol.info[key]
            else:
                # Add vacuum property with prefix if not in environment
                merged_mol.info[PREFIX_VACUUM + key] = vac_mol.info[key]
        
        # Add environment-only properties with prefix
        for key in env_mol.info:
            if key not in COMMON_KEYS and key not in vac_mol.info:
                merged_mol.info[PREFIX_ENV + key] = env_mol.info[key]
        
        # Process arrays
        for key in vac_mol.arrays:
            if key in COMMON_KEYS:
                # Verify common arrays match
                assert np.allclose(vac_mol.arrays[key], env_mol.arrays[key]), f"Mismatch in {key} for structure {i}"
            elif key in ESP_KEYS:
                # Skip ESP-related arrays for vacuum
                continue
            elif key in DIFF_KEYS and key in env_mol.arrays:
                # Calculate and store differences for array properties
                diff_array = env_mol.arrays[key] - vac_mol.arrays[key]
                merged_mol.set_array(DIFF_PREFIX + key, diff_array)
                
                # Store arrays with prefixes
                merged_mol.set_array(PREFIX_VACUUM + key, vac_mol.arrays[key])
                merged_mol.set_array(PREFIX_ENV + key, env_mol.arrays[key])
                
                # Remove original array
                if key in merged_mol.arrays:
                    del merged_mol.arrays[key]
            else:
                # Add vacuum array with prefix if not in environment
                merged_mol.set_array(PREFIX_VACUUM + key, vac_mol.arrays[key])
        
        # Add environment-only arrays with prefix
        for key in env_mol.arrays:
            if key not in COMMON_KEYS and key not in vac_mol.arrays and key not in merged_mol.arrays:
                merged_mol.set_array(PREFIX_ENV + key, env_mol.arrays[key])
        
        merged_molecules.append(merged_mol)
    
    return merged_molecules

def main() -> None:
    args: argparse.Namespace = parse_args()
    config_data: Dict[str, Any] = read_config(args)

    vacuum_file = config_data["VACUUM_FILE"]
    environment_file = config_data["ENVIRONMENT_FILE"]
    output_file = config_data["OUTFILE"]
    tolerance = config_data["TOLERANCE"]
    
    print(f"Reading vacuum file: {vacuum_file}")
    vacuum_molecules = read_extxyz(vacuum_file)
    print(f"Found {len(vacuum_molecules)} structures in vacuum file")

    print(f"Reading environment file: {environment_file}")
    environment_molecules = read_extxyz(environment_file)
    print(f"Found {len(environment_molecules)} structures in environment file")

    print("Checking structural identity...")
    check_structural_identity(vacuum_molecules, environment_molecules, tolerance)
    
    print("Merging properties...")
    merged_molecules = merge_molecules(vacuum_molecules, environment_molecules)
    
    print(f"Writing merged file to: {output_file}")
    write(output_file, merged_molecules, format="extxyz")
    
    print("Done!")

if __name__ == "__main__":
    main()