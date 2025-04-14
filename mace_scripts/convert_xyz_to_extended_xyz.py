#!/usr/bin/env python3
import argparse
import json
import os
import numpy as np
from typing import Tuple, Generator, Sequence, Dict

from ase import Atoms

# Default config values
#DATA_FOLDER: str = "/lustre/work/ws/ws1/ka_he8978-dipeptide/training_data/B3LYP_aug-cc-pVTZ_water"
#DATA_FOLDER: str = "/lustre/work/ws/ws1/ka_he8978-thiol_disulfide/training_data/B3LYP_aug-cc-pVTZ_vacuum"
DATA_FOLDER: str = os.getcwd()
GEOMETRY_FILE: str = "ThiolDisulfidExchange.xyz" # Angstrom units to Angstrom units
ENERGY_FILE: str = "energies.txt" # Hartree to eV
FORCE_FILE: str = "forces_conv.xyz" # Hartree/Bohr to eV/Angstrom, xyz format, assumed to be not actually forces but gradients, transformed to forces by multiplying by -1
CHARGE_FILE: str = "charges.txt" # e to e
ESP_FILE: str = "esps_by_mm.txt" # eV/e to eV/e, optional, gets filled with zeros if not present
ESP_GRAD_FILE: str = "esp_gradients_conv.xyz" # eV/e/B to eV/e/A, xyz format, optional, gets filled with zeros if not present (could be transformed to electric field by multiplying by -1)
DIPOLE_FILE: str = "dipoles.txt" # au to au, optional, gets filled with zeros if not present
QUADRUPOLE_FILE: str = "quadrupoles.txt" # au to au, optional, gets filled with zeros if not present
OUTFILE: str = "geoms.extxyz"

BOXSIZE: float = 22.0 # nm to Angstrom, assuming cubic box, not individual per geometry so far, TODO: read from input file

# Property keys in .extxyz file, are expected like this in mace scripts
energy_key: str = "ref_energy"
force_key: str = "ref_force"
charge_key: str = "ref_charge"
esp_key: str = "esp"
esp_gradient_key: str = "esp_gradient"
total_charge_key: str = "total_charge"
dipole_key: str = "ref_dipole"
quadrupole_key: str = "ref_quadrupole"

# Conversion factors
H_to_eV = 27.211386245988
H_B_to_eV_A = 51.422086190832
e_to_e = 1.0
nm_to_A = 10.0

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Give config file")
    ap.add_argument("-c", "--conf", default=None, type=str, dest="config_path", action="store", required=False, help="Path to config file, default: None", metavar="config")
    ap.add_argument("-g", "--gpuid", default=None, type=int, required=False) # Just here as a dummy, nothing actually uses a GPU, other scripts got submitted with --gpuid
    args = ap.parse_args()
    return args

def read_config(args: argparse.Namespace) -> dict:
    config_path = args.config_path

    if config_path is not None:
        try:
            with open(config_path, 'r') as config_file:
                config_data = json.load(config_file)
        except FileNotFoundError:
            print(f"Config file {config_path} not found.")
            exit(1)

    # Set defaults to global defaults, otherwise use config file values
    config_data.setdefault("DATA_FOLDER", DATA_FOLDER)
    config_data.setdefault("GEOMETRY_FILE", GEOMETRY_FILE)
    config_data.setdefault("ENERGY_FILE", ENERGY_FILE)
    config_data.setdefault("FORCE_FILE", FORCE_FILE)
    config_data.setdefault("CHARGE_FILE", CHARGE_FILE)
    config_data.setdefault("ESP_FILE", ESP_FILE)
    config_data.setdefault("ESP_GRAD_FILE", ESP_GRAD_FILE)
    config_data.setdefault("DIPOLE_FILE", DIPOLE_FILE)
    config_data.setdefault("QUADRUPOLE_FILE", QUADRUPOLE_FILE)
    config_data.setdefault("OUTFILE", OUTFILE)
    config_data.setdefault("BOXSIZE", BOXSIZE)

    TOTAL_CHARGE = config_data.get("TOTAL_CHARGE", None)
    if TOTAL_CHARGE is not None:
        print("INFO: Giving total charge as directly as input is deprecated. Using charge-file instead.")

    return config_data

def read_xyz(filename: str) -> Generator[int, str, Atoms]:
    with open(filename, 'r') as file:
        while True:
            try:
                n_atoms = int(file.readline())
                comment = file.readline().strip()
                atoms = [file.readline().strip().split() for _ in range(n_atoms)]
                yield n_atoms, comment, atoms
            except ValueError:  # End of file
                break

def load_energy_data(energy_file: str) -> np.ndarray:
    energies = np.loadtxt(energy_file)
    return energies

def load_charge_data(charge_file: str) -> Tuple[Sequence[np.ndarray], np.ndarray]:
    charges = []
    with open(charge_file, 'r') as file:
        for row in file:
            charges.append(np.array(row.strip().split(), dtype=float))
    
    total_charges = np.array([np.sum(charge_array) for charge_array in charges])
    return charges, total_charges

def load_force_data(force_file: str) -> Sequence[np.ndarray]:
    forces = []
    with open(force_file, 'r') as file:
        while True:
            try:
                n_atoms = int(file.readline())
                file.readline()  # Skip comment
                forces.append(np.array([file.readline().strip().split()[-3:] for _ in range(n_atoms)], dtype=float))
            except ValueError:  # End of file
                break
    return forces

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

def write_extxyz(
        config_data: Dict[str, str|int|float],
        molecules: Sequence[Tuple[int, str, Atoms]],
        energies: np.ndarray,
        forces: Sequence[np.ndarray],
        charges: Sequence[np.ndarray],
        total_charges: np.ndarray,
        esps: Sequence[np.ndarray],
        electric_fields: Sequence[np.ndarray],
        dipoles: np.ndarray,
        quadrupoles: np.ndarray,
    ) -> None:
    outfile = config_data["OUTFILE"]
    boxsize = config_data["BOXSIZE"]*nm_to_A 
    lattice_vector = f'{boxsize:0.1f} 0.0 0.0 0.0 {boxsize:0.1f} 0.0 0.0 0.0 {boxsize:0.1f}'
    file = open(outfile, 'w')
    for mol_idx in range(len(molecules)):
        n_atoms, comment, atoms = molecules[mol_idx]
        file.write(f"{n_atoms}\n")
        info_line = (
            f'Lattice="{lattice_vector}" '
            f'Properties=species:S:1:pos:R:3:{force_key}:R:3:{charge_key}:R:1:{esp_key}:R:1:{esp_gradient_key}:R:3 '
            f'{energy_key}={energies[mol_idx]} '
            f'{total_charge_key}={total_charges[mol_idx]:1.1f} '
            f'{dipole_key}="{dipoles[mol_idx,0]:1.5f} {dipoles[mol_idx,1]:1.5f} {dipoles[mol_idx,2]:1.5f}" '
            f'{quadrupole_key}="{quadrupoles[mol_idx,0]:1.5f} {quadrupoles[mol_idx,1]:1.5f} {quadrupoles[mol_idx,2]:1.5f} '
            f'{quadrupoles[mol_idx,3]:1.5f} {quadrupoles[mol_idx,4]:1.5f} {quadrupoles[mol_idx,5]:1.5f}" '
            f'pbc="F F F" comment="{comment}"\n'
        )
        file.write(info_line)
        for at_idx, atom in enumerate(atoms):
            atom_line = " ".join(atom[:4])  # Assuming atom format is [element, x, y, z]
            force_line = " ".join(map(lambda x: f"{x: .8f}", forces[mol_idx][at_idx]))
            charge = f"{charges[mol_idx][at_idx]: .6f}"
            esp = f"{esps[mol_idx][at_idx]: .6f}"
            gradient = " ".join(map(lambda x: f"{x: .6f}", electric_fields[mol_idx][at_idx]))
            file.write(f"{atom_line} {force_line} {charge} {esp} {gradient}\n")
    file.close()

def main() -> None:
    args: argparse.Namespace = parse_args()
    config_data: Dict[str, str|int|float] = read_config(args)

    data_folder             = config_data["DATA_FOLDER"]
    geom_file               = os.path.join(data_folder, config_data["GEOMETRY_FILE"])
    energy_file             = os.path.join(data_folder, config_data["ENERGY_FILE"])
    force_file              = os.path.join(data_folder, config_data["FORCE_FILE"])
    charge_file             = os.path.join(data_folder, config_data["CHARGE_FILE"])
    esp_file                = os.path.join(data_folder, config_data["ESP_FILE"])
    esp_gradient_file       = os.path.join(data_folder, config_data["ESP_GRAD_FILE"])
    dipole_file             = os.path.join(data_folder, config_data["DIPOLE_FILE"])
    quadrupole_file         = os.path.join(data_folder, config_data["QUADRUPOLE_FILE"])

    # Read data
    molecules: Sequence[Tuple[int, str, Atoms]] = list(read_xyz(geom_file)) # List of tuples (n_atoms, comment, Atoms)
    energies: np.ndarray = load_energy_data(energy_file)
    charges: Sequence[np.ndarray] # Possibly ragged list
    total_charges: Sequence[np.ndarray]
    charges, total_charges = load_charge_data(charge_file) 
    forces: Sequence[np.ndarray] = load_force_data(force_file) # Possibly ragged list
    esps: Sequence[np.ndarray] # Possibly ragged list
    gradients: Sequence[np.ndarray] # Possibly ragged list
    if os.path.exists(esp_file) and os.path.exists(esp_gradient_file):
        print("ESP files found")
        esps, gradients = load_esp_data(esp_file, esp_gradient_file)
    else:
        print("ESP files not found, filling with zeros")
        esps = [np.zeros(len(molecule[2])) for molecule in molecules]
        gradients = [np.zeros((len(molecule[2]), 3)) for molecule in molecules]
    if os.path.exists(dipole_file):
        print("Dipole file found")
        dipoles: np.ndarray = load_dipole_data(dipole_file)
    else:
        print("Dipole file not found, filling with zeros")
        dipoles = np.zeros((len(molecules), 3))
    if os.path.exists(quadrupole_file):
        print("Quadrupole file found")
        quadrupoles: np.ndarray = load_dipole_data(quadrupole_file)
    else:
        print("Quadrupole file not found, filling with zeros")
        quadrupoles = np.zeros((len(molecules), 6))

    # Check and assert data
    assert len(molecules) == len(energies), f"Number of geometries ({len(molecules)}) does not match number of energies ({len(energies)})"
    assert len(molecules) == len(charges), f"Number of geometries ({len(molecules)}) does not match number of charge arrays ({len(charges)})"
    assert len(molecules) == len(total_charges), f"Number of geometries ({len(molecules)}) does not match number of total charges ({len(total_charges)})"
    assert len(molecules) == len(forces), f"Number of geometries ({len(molecules)}) does not match number of force arrays ({len(forces)})"
    assert len(molecules) == len(esps), f"Number of geometries ({len(molecules)}) does not match number of ESP arrays ({len(esps)})"
    assert len(molecules) == len(gradients), f"Number of geometries ({len(molecules)}) does not match number of ESP gradient arrays ({len(gradients)})"
    assert len(molecules) == len(dipoles), f"Number of geometries ({len(molecules)}) does not match number of dipoles ({len(dipoles)})"
    assert len(molecules) == len(quadrupoles), f"Number of geometries ({len(molecules)}) does not match number of quadrupoles ({len(quadrupoles)})"

    # Convert units
    energies *= H_to_eV
    forces = [force_matrix * H_B_to_eV_A * -1 for force_matrix in forces]
    esps = [esp_array for esp_array in esps]
    esp_gradients = [gradient_matrix for gradient_matrix in gradients]

    write_extxyz(config_data, molecules, energies, forces, charges, total_charges, esps, esp_gradients, dipoles, quadrupoles)

if __name__ == "__main__":
    main()

    