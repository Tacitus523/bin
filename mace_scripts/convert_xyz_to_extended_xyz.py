#!/usr/bin/env python3
import argparse
import json
import os
import numpy as np
from typing import Tuple, Generator, Sequence, Dict, List, Optional

from ase import Atoms
from ase.io import read, write

# Default config values
DATA_FOLDER: str = os.getcwd()
GEOMETRY_FILE: str = "ThiolDisulfidExchange.xyz" # Angstrom units to Angstrom units
ENERGY_FILE: str = "energies.txt" # Hartree to eV
GRADIENT_FILE: str = "gradients.xyz" # Hartree/Bohr to eV/Angstrom, xyz format, energy gradients not forces, transformed to forces by multiplying by -1
CHARGE_FILE: Optional[str] = "charges.txt" # e to e, optional, gets filled with zeros if not present
ESP_FILE: Optional[str] = "esps_by_mm.txt" # eV/e to eV/e, optional, gets filled with zeros if not present
ESP_GRAD_FILE: Optional[str] = "esp_gradients_conv.xyz" # eV/e/B to eV/e/A, xyz format, optional, gets filled with zeros if not present (could be transformed to electric field by multiplying by -1)
DIPOLE_FILE: Optional[str] = "dipoles.txt" # au to Debye, optional, gets filled with zeros if not present
QUADRUPOLE_FILE: Optional[str] = "quadrupoles.txt" # au to au, optional, gets filled with zeros if not present
PC_FILE: Optional[str] = "mm_data.pc" # Angstrom units to Angstrom units, optional
OUTFILE: Optional[str] = "geoms.extxyz"

TOTAL_CHARGE: Optional[float] = 0.0 # Mostly deprecated, charge_file used instead, used if charge_file is not present

BOXSIZE: Optional[float] = 22.0 # nm to Angstrom, assuming cubic box, not individual per geometry so far, TODO: read from input file

FORMAT: Optional[str] = "mace" # or "fieldmace", slight differences in the output format

# Property keys in .extxyz file, are expected like this in mace scripts
energy_key: str = "ref_energy"
force_key: str = "ref_force"
charge_key: str = "ref_charge"
esp_key: str = "esp"
esp_gradient_key: str = "esp_gradient"
total_charge_key: str = "total_charge"
dipole_key: str = "ref_dipole"
quadrupole_key: str = "ref_quadrupole"
mm_positions_key: str = "mm_positions"
mm_charges_key: str = "mm_charges"

# Conversion factors
H_to_eV: float = 27.211386245988
H_B_to_eV_A: float = 51.422086190832
e_to_e: float = 1.0
nm_to_A: float = 10.0
debye_to_eA: float = 0.2081943
eA_to_debye: float = 1.0 / debye_to_eA
debye_to_ea0: float = 0.3934303
ea0_to_debye: float = 1.0 / debye_to_ea0

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Give config file")
    ap.add_argument("-c", "--conf", default=None, type=str, dest="config_path", action="store", required=False, help="Path to config file, default: None", metavar="config")
    ap.add_argument("-f", "--format", default=None, type=str, choices=["mace", "fieldmace", None], required=False, help="Format of the output file, default: %s" % FORMAT, metavar="format")
    ap.add_argument("-m", "--max_mm", default=None, type=int, required=False, help="Max number of MM atoms per molecule, default: None", metavar="max_mm")
    args = ap.parse_args()
    return args

def read_config(args: argparse.Namespace) -> Dict[str, str|int|float]:
    config_path = args.config_path

    if config_path is not None:
        try:
            with open(config_path, 'r') as config_file:
                config_data = json.load(config_file)
        except FileNotFoundError:
            print(f"Config file {config_path} not found.")
            exit(1)

    # Set default values
    config_data.setdefault("DATA_FOLDER", DATA_FOLDER)
    config_data.setdefault("GEOMETRY_FILE", GEOMETRY_FILE)
    config_data.setdefault("ENERGY_FILE", ENERGY_FILE)
    config_data.setdefault("GRADIENT_FILE", GRADIENT_FILE)
    config_data.setdefault("CHARGE_FILE", CHARGE_FILE)
    config_data.setdefault("ESP_FILE", ESP_FILE)
    config_data.setdefault("ESP_GRAD_FILE", ESP_GRAD_FILE)
    config_data.setdefault("DIPOLE_FILE", DIPOLE_FILE)
    config_data.setdefault("QUADRUPOLE_FILE", QUADRUPOLE_FILE)
    config_data.setdefault("PC_FILE", PC_FILE)
    config_data.setdefault("OUTFILE", OUTFILE)
    config_data.setdefault("TOTAL_CHARGE", TOTAL_CHARGE)
    config_data.setdefault("BOXSIZE", BOXSIZE)
    config_data.setdefault("FORMAT", FORMAT)

    # Set config values from command line arguments, otherwise use config file values
    if args.format is not None:
        config_data["FORMAT"] = args.format
    else:
        config_data.setdefault("FORMAT", args.format)
    if args.max_mm is not None:
        config_data["MAX_MM"] = args.max_mm
    else:
        config_data.setdefault("MAX_MM", args.max_mm)
    
    if "FORCE_FILE" in config_data:
        print("WARNING: FORCE_FILE is deprecated, use GRADIENT_FILE instead")
        config_data["GRADIENT_FILE"] = config_data["FORCE_FILE"]

    return config_data

def read_xyz(filename: str) -> List[Atoms]:
    molecules = read(filename, index=":")
    if isinstance(molecules, Atoms):
        molecules = [molecules]
    elif not isinstance(molecules, list):
        raise ValueError(f"Unexpected type for molecules: {type(molecules)}")
    return molecules

def load_energy_data(energy_file: str) -> np.ndarray:
    energies = np.loadtxt(energy_file)
    return energies

def load_charge_data(charge_file: str) -> Tuple[Sequence[np.ndarray], np.ndarray]:
    charges = []
    with open(charge_file, 'r') as file:
        for row in file:
            charges.append(np.array(row.strip().split(), dtype=float))
    
    total_charges = np.round([np.sum(charge_array) for charge_array in charges], 1)
    return charges, total_charges

def load_gradient_data(gradient_file: str) -> Sequence[np.ndarray]:
    gradients = []
    with open(gradient_file, 'r') as file:
        while True:
            try:
                n_atoms = int(file.readline())
                file.readline()  # Skip comment
                gradients.append(np.array([file.readline().strip().split()[-3:] for _ in range(n_atoms)], dtype=float))
            except ValueError:  # End of file
                break
    return gradients

def load_esp_data(esp_file: str, esp_gradient_file: str) -> Tuple[Sequence[np.ndarray], Sequence[np.ndarray]]:
    esps = []
    with open(esp_file, 'r') as file:
        for row in file:
            esps.append(np.array(row.strip().split(), dtype=float))

    gradients = []
    with open(esp_gradient_file, 'r') as file:
        while True:
            try:
                n_atoms = int(file.readline())
                file.readline()  # Skip comment
                gradients.append(np.array([file.readline().strip().split()[-3:] for _ in range(n_atoms)], dtype=float))
            except ValueError:  # End of file
                break
    return esps, gradients

def load_dipole_data(dipole_file: str) -> np.ndarray:
    dipoles = np.loadtxt(dipole_file)
    return dipoles

def load_quadrupole_data(quadrupole_file: str) -> np.ndarray:
    quadrupoles = np.loadtxt(quadrupole_file)
    return quadrupoles

def find_max_chunk_size(filename: str) -> int:
    """
    Find the maximum chunk size in a file where each chunk starts with a size indicator.
    
    Args:
        filename (str): Path to the file to analyze
        
    Returns:
        int: Maximum chunk size found in the file
    """
    max_size = 0
    
    if not os.path.exists(filename):
        return max_size

    with open(filename, 'r') as file:
        while True:
            # Try to read the chunk size
            size_line = file.readline().strip()
            if not size_line:  # End of file
                break
                
            try:
                chunk_size = int(size_line)
                max_size = max(max_size, chunk_size)
                
                # Skip the chunk data to get to the next size indicator
                for _ in range(chunk_size):
                    if not file.readline():  # Unexpected end of file
                        break
                        
            except ValueError:
                # Not a size indicator line, skip
                continue
    
    return max_size

def validate_chunk_sizes(filename: str, expected_max_size: int, threshold: int = 100) -> List[Tuple[int, int]]:
    """
    Validate chunk sizes in a file against an expected maximum size and identify outliers.
    
    Args:
        filename (str): Path to the file to analyze
        expected_max_size (int): Expected maximum chunk size to check against
        threshold (int): Threshold for determining significant differences (default: 100)
        
    Returns:
        List[Tuple[int, int]]: List of tuples (chunk_index, chunk_size) for chunks with significant size differences
    """
    outliers = []
    chunk_index = 0
    
    with open(filename, 'r') as file:
        while True:
            # Try to read the chunk size
            size_line = file.readline().strip()
            if not size_line:  # End of file
                break

            try:
                chunk_size = int(size_line)
            except ValueError:
                # Not a size indicator line, skip
                continue


            
            # Check if the chunk size differs significantly from the expected max
            if abs(chunk_size - expected_max_size) >= threshold:
                outliers.append((chunk_index, chunk_size))
                # print(f"Warning: Chunk {chunk_index} has size {chunk_size}, which differs "
                #         f"from expected max size {expected_max_size} by {abs(chunk_size - expected_max_size)}")
            
            # Skip the chunk data to get to the next size indicator
            for _ in range(chunk_size):
                if not file.readline():  # Unexpected end of file
                    break
            
            chunk_index += 1
                        
    
    if outliers:
        outlier_percentage = (len(outliers) / chunk_index) * 100
        print(f"Found {len(outliers)} outliers out of {chunk_index} chunks ({outlier_percentage:.2f}%)")
        print("This may bias the training of the model, because missing data gets filled with zeros.")
        print("Consider creating separate systems for different sizes.")
    else:
        print(f"All {chunk_index} chunks are within {threshold} of the expected maximum size {expected_max_size}")
        
    return outliers

def read_chunked_file(filename: str) -> List[np.ndarray]:
    """
    Read a file in chunks where each chunk starts with a size indicator.
    Returns a list of numpy arrays for each chunk based on the specified size.
    
    Args:
        filename (str): Path to the file to read
        
    Returns:
        List[numpy.ndarray]: List of arrays, each with shape (chunk_size, *) for each chunk
    """
    result = []
    with open(filename, 'r') as file:
        while True:
            # Try to read the chunk size
            size_line = file.readline().strip()
            if not size_line:  # End of file
                break
                
            try:
                chunk_size = int(size_line)
            except ValueError:
                # Not a size indicator line, skip
                continue
                
            # Read the specified number of lines
            chunk_data = []
            for _ in range(chunk_size):
                line = file.readline()
                if not line:  # Unexpected end of file
                    break
                    
                values = line.strip().split()
                chunk_data.append(values)

            if chunk_data:
                result.append(np.array(chunk_data, dtype=float))

    return result

def pad_arrays(arrays: List[np.ndarray], target_length: int) -> np.ndarray:
    """
    Pad numpy arrays to a target length with zeros along their current axis 0. Expected shape is (n_mm_atoms, m).
    
    Args:
        array (numpy.ndarray): Array to pad
        target_length (int): Target length for padding
        
    Returns:
        numpy.ndarray: Stacked arrays along axis 0, padded to the target length. Return shape (n_mm_molecules, max_n_mm_atoms, m)
    """
    padded_arrays = []
    for array in arrays:
        pad_width = [(0, target_length - array.shape[0])] + [(0, 0)] * (len(array.shape) - 1)
        padded_array = np.pad(array, pad_width, mode='constant', constant_values=0)
        padded_arrays.append(padded_array)
    
    return np.stack(padded_arrays, axis=0)

def write_extxyz(
        config_data: Dict[str, str|int|float],
        molecules: List[Atoms],
        energies: np.ndarray,
        forces: Sequence[np.ndarray],
        charges: Sequence[np.ndarray],
        total_charges: np.ndarray,
        esps: Sequence[np.ndarray],
        esp_gradients: Sequence[np.ndarray],
        dipoles: np.ndarray,
        quadrupoles: np.ndarray,
    ) -> None:
    outfile = config_data["OUTFILE"]
    boxsize = config_data["BOXSIZE"]
    
    edited_molecules: List[Atoms] = []
    for mol_idx in range(len(molecules)):
        molecule: Atoms = molecules[mol_idx]
        molecule.set_cell([boxsize, boxsize, boxsize], scale_atoms=False)
        molecule.set_pbc((False, False, False))

        molecule.info[energy_key] = energies[mol_idx]
        molecule.info[total_charge_key] = total_charges[mol_idx]
        molecule.info[dipole_key] = dipoles[mol_idx]
        molecule.info[quadrupole_key] = quadrupoles[mol_idx]

        molecule.set_array(force_key, forces[mol_idx])
        molecule.set_array(charge_key, charges[mol_idx])
        molecule.set_array(esp_key, esps[mol_idx])
        molecule.set_array(esp_gradient_key, esp_gradients[mol_idx])

        edited_molecules.append(molecule)
    write(outfile, edited_molecules, format="extxyz", append=False)

def write_fieldmace_extxyz(
        config_data: Dict[str, str|int|float],
        molecules: List[Atoms],
        energies: np.ndarray,
        forces: Sequence[np.ndarray],
        charges: Sequence[np.ndarray],
        total_charges: np.ndarray,
        esps: Sequence[np.ndarray],
        esp_gradients: Sequence[np.ndarray],
        dipoles: np.ndarray,
        mm_charges: np.ndarray,
        mm_positions: np.ndarray,
    ) -> None:
    outfile = config_data["OUTFILE"]
    boxsize = config_data["BOXSIZE"]
    
    if energies.ndim == 1:
        n_states = 1
    else:
        n_states = energies.shape[1]

    edited_molecules: List[Atoms] = []
    for mol_idx in range(len(molecules)):
        molecule: Atoms = molecules[mol_idx]

        energy = energies[mol_idx].reshape(1, n_states)
        force = forces[mol_idx].reshape(-1, n_states, 3)
        dipole = dipoles[mol_idx].reshape(n_states, 3)

        # Format MM charges as a space-separated string
        mm_charges_str = " ".join([f"{charge:.4f}" for charge in mm_charges[mol_idx]])
        
        molecule.set_cell([boxsize, boxsize, boxsize], scale_atoms=False)
        molecule.set_pbc((False, False, False))
        molecule.info["n_states"] = n_states
        molecule.info[energy_key] = energy
        molecule.info[force_key] = force
        molecule.info[total_charge_key] = total_charges[mol_idx]
        molecule.info[dipole_key] = dipole
        molecule.info[mm_positions_key] = mm_positions[mol_idx]
        molecule.info[mm_charges_key] = mm_charges_str

        molecule.set_array(charge_key, charges[mol_idx])
        molecule.set_array(esp_key, esps[mol_idx])
        molecule.set_array(esp_gradient_key, esp_gradients[mol_idx])
        
        edited_molecules.append(molecule)

    write(outfile, edited_molecules, format="extxyz", append=False)

def main() -> None:
    args: argparse.Namespace = parse_args()
    config_data: Dict[str, str|int|float] = read_config(args)

    data_folder             = config_data["DATA_FOLDER"]
    geom_file               = os.path.join(data_folder, config_data["GEOMETRY_FILE"])
    energy_file             = os.path.join(data_folder, config_data["ENERGY_FILE"])
    gradient_file           = os.path.join(data_folder, config_data["GRADIENT_FILE"])
    charge_file             = os.path.join(data_folder, config_data["CHARGE_FILE"])
    esp_file                = os.path.join(data_folder, config_data["ESP_FILE"])
    esp_gradient_file       = os.path.join(data_folder, config_data["ESP_GRAD_FILE"])
    dipole_file             = os.path.join(data_folder, config_data["DIPOLE_FILE"])
    quadrupole_file         = os.path.join(data_folder, config_data["QUADRUPOLE_FILE"])
    point_charges_file_path = os.path.join(data_folder, config_data["PC_FILE"])
    extxyz_format           = config_data["FORMAT"]
    print(f"Using format: {extxyz_format}")

    # Read data
    molecules: List[Atoms] = read_xyz(geom_file)
    energies: np.ndarray = load_energy_data(energy_file) 

    charges: Sequence[np.ndarray] # Possibly ragged list
    total_charges: Sequence[np.ndarray]
    if os.path.exists(charge_file):
        print("Charge file found")
        charges, total_charges = load_charge_data(charge_file)
    else:
        print("Charge file not found, filling with zeros, using total charge: %s" % config_data["TOTAL_CHARGE"])
        print("Warning: Unable to differentiate between different total charges")
        charges = [np.zeros(len(molecule)) for molecule in molecules]
        total_charges = np.zeros(len(molecules)) + config_data["TOTAL_CHARGE"]

    gradients: Sequence[np.ndarray] = load_gradient_data(gradient_file) # Possibly ragged list

    esps: Sequence[np.ndarray] # Possibly ragged list
    esp_gradients: Sequence[np.ndarray] # Possibly ragged list
    if os.path.exists(esp_file) and os.path.exists(esp_gradient_file):
        print("ESP files found")
        esps, esp_gradients = load_esp_data(esp_file, esp_gradient_file)
    else:
        print("ESP files not found, filling with zeros")
        esps = [np.zeros(len(molecule)) for molecule in molecules]
        esp_gradients = [np.zeros((len(molecule), 3)) for molecule in molecules]

    dipoles: np.ndarray
    if os.path.exists(dipole_file):
        print("Dipole file found")
        dipoles = load_dipole_data(dipole_file)
    else:
        print("Dipole file not found, filling with zeros")
        dipoles = np.zeros((len(molecules), 3))

    quadrupoles: np.ndarray
    if os.path.exists(quadrupole_file):
        print("Quadrupole file found")
        quadrupoles = load_dipole_data(quadrupole_file)
    else:
        print("Quadrupole file not found, filling with zeros")
        quadrupoles = np.zeros((len(molecules), 6))

    if extxyz_format == "fieldmace":
        mm_charges: np.ndarray
        mm_positions: np.ndarray
        if os.path.exists(point_charges_file_path):
            print(f"Point charges file found")
            # Collect the data from the point charges files
            max_mm_molecules: int = find_max_chunk_size(point_charges_file_path)
            print(f"Max chunk size in {point_charges_file_path}: {max_mm_molecules}")
            if config_data["MAX_MM"] is not None:
                if config_data["MAX_MM"] < max_mm_molecules:
                    raise ValueError(f"Max number of MM atoms per molecule ({config_data['MAX_MM']}) is smaller than the max chunk size in the point charges file ({max_mm_molecules}).")
                elif config_data["MAX_MM"] > max_mm_molecules:
                    print(f"Warning: Max number of MM atoms per molecule ({config_data['MAX_MM']}) is larger than the max chunk size in the point charges file ({max_mm_molecules}).")
                max_mm_molecules = config_data["MAX_MM"]
                
            validate_chunk_sizes(point_charges_file_path, max_mm_molecules, threshold=100)

            point_charges_contents: List[np.ndarray] = read_chunked_file(point_charges_file_path) # Shape (n_mm_molecules, n_mm_atoms, 4), possibly irregular
            point_charges_contents: np.ndarray = pad_arrays(point_charges_contents, max_mm_molecules)
            mm_charges     = point_charges_contents[:, :, 0]
            mm_positions   = point_charges_contents[:, :, 1:4]
        else:
            print("Point charges file not found, filling with zeros")
            # Create dummy data for MM charges and coordinates
            mm_charges = np.zeros((len(molecules), 1))
            mm_positions = np.zeros((len(molecules), 1, 3))

    # Check and assert data
    assert len(molecules) == len(energies), f"Number of geometries ({len(molecules)}) does not match number of energies ({len(energies)})"
    assert len(molecules) == len(charges), f"Number of geometries ({len(molecules)}) does not match number of charge arrays ({len(charges)})"
    assert len(molecules) == len(total_charges), f"Number of geometries ({len(molecules)}) does not match number of total charges ({len(total_charges)})"
    assert len(molecules) == len(gradients), f"Number of geometries ({len(molecules)}) does not match number of force arrays ({len(forces)})"
    assert len(molecules) == len(esps), f"Number of geometries ({len(molecules)}) does not match number of ESP arrays ({len(esps)})"
    assert len(molecules) == len(esp_gradients), f"Number of geometries ({len(molecules)}) does not match number of ESP gradient arrays ({len(esp_gradients)})"
    assert len(molecules) == len(dipoles), f"Number of geometries ({len(molecules)}) does not match number of dipoles ({len(dipoles)})"
    assert len(molecules) == len(quadrupoles), f"Number of geometries ({len(molecules)}) does not match number of quadrupoles ({len(quadrupoles)})"
    if extxyz_format == "fieldmace":
        assert len(molecules) == len(mm_charges), f"Number of geometries ({len(molecules)}) does not match number of MM charges ({len(mm_charges)})"
        assert len(molecules) == len(mm_positions), f"Number of geometries ({len(molecules)}) does not match number of MM coordinates ({len(mm_positions)})"

    # Convert units
    config_data["BOXSIZE"] = config_data["BOXSIZE"]*nm_to_A 
    energies *= H_to_eV
    forces = [gradient_matrix * H_B_to_eV_A * -1 for gradient_matrix in gradients]
    esps = [esp_array for esp_array in esps]
    esp_gradients = [gradient_matrix for gradient_matrix in esp_gradients]
    dipoles *= ea0_to_debye

    if extxyz_format == "mace":
        write_extxyz(config_data, molecules, energies, forces, charges, total_charges, esps, esp_gradients, dipoles, quadrupoles)
    elif extxyz_format == "fieldmace":
        write_fieldmace_extxyz(config_data, molecules, energies, forces, charges, total_charges, esps, esp_gradients, dipoles, mm_charges, mm_positions)
    else:   
        raise ValueError(f"Unknown format: {extxyz_format}")
    print(f"Output written to {config_data['OUTFILE']}")

if __name__ == "__main__":
    main()