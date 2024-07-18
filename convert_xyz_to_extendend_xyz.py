#!/usr/bin/env python3
import os
import numpy as np
from typing import Tuple, Generator, Sequence

from ase import Atoms

DATA_FOLDER: str = "/lustre/work/ws/ws1/ka_he8978-dipeptide/training_data/B3LYP_aug-cc-pVTZ_water"
GEOM_FILE: str = "geoms.xyz" # Angstrom units to Angstrom units
ENERGY_FILE: str = "energies.txt" # Hartree to eV
FORCE_FILE: str = "forces_conv.xyz" # Hartree/Bohr to eV/Angstrom, xyz format, assumed to be not actually forces but gradients, transformed to forces by multiplying by -1
CHARGE_FILE: str = "charges.txt" # e to e
ESP_FILE: str = "esps_by_mm.txt" # H/e to eV/e, optional, gets filled with zeros if not present
ESP_GRADIENT_FILE: str = "esp_gradients_conv.xyz" # H/e/B to eV/e/A, xyz format, transformed to electric field by multiplying by -1, optional, gets filled with zeros if not present
OUTFILE: str = "geoms.extxyz"

TOTAL_CHARGE: float = 0.0 # e to e
BOXSIZE: float = 3.0 # nm to Angstrom, assuming cubic box, irrelevant unless periodic system

H_to_eV = 27.211386245988
H_B_to_eV_A = 51.422086190832
e_to_e = 1.0
nm_to_A = 10.0

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

def load_energy_data(energy_file):
    energies = np.loadtxt(energy_file)
    return energies

def load_charge_data(charge_file):
    charges = []
    with open(charge_file, 'r') as file:
        for row in file:
            charges.append(np.array(row.strip().split(), dtype=float))
    return charges

def load_force_data(force_file):
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

def load_esp_data(esp_file, esp_gradient_file):
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

def write_extxyz(outfile, molecules, energies, forces, charges, esps, electric_fields):
    boxsize = BOXSIZE*nm_to_A 
    lattice_vector = f'{boxsize:0.1f} 0.0 0.0 0.0 {boxsize:0.1f} 0.0 0.0 0.0 {boxsize:0.1f}'
    with open(outfile, 'w') as file:
        for mol_idx, (n_atoms, comment, atoms) in enumerate(molecules):
            file.write(f"{n_atoms}\n")
            file.write(f'Lattice="{lattice_vector}" Properties=species:S:1:pos:R:3:ref_force:R:3:ref_charge:R:1:esp:R:1:electric_field:R:3 ref_energy={energies[mol_idx]} total_charge={TOTAL_CHARGE} pbc="F F F" comment="{comment}"\n')
            for at_idx, atom in enumerate(atoms):
                atom_line = " ".join(atom[:4])  # Assuming atom format is [element, x, y, z]
                force_line = " ".join(map(lambda x: f"{x: .8f}", forces[mol_idx][at_idx]))
                charge = f"{charges[mol_idx][at_idx]: .6f}"
                esp = f"{esps[mol_idx][at_idx]: .6f}"
                gradient = " ".join(map(lambda x: f"{x: .6f}", electric_fields[mol_idx][at_idx]))
                file.write(f"{atom_line} {force_line} {charge} {esp} {gradient}\n")

def main():
    geom_file = os.path.join(DATA_FOLDER, GEOM_FILE)
    energy_file = os.path.join(DATA_FOLDER, ENERGY_FILE)
    force_file = os.path.join(DATA_FOLDER, FORCE_FILE)
    charge_file = os.path.join(DATA_FOLDER, CHARGE_FILE)
    esp_file = os.path.join(DATA_FOLDER, ESP_FILE)
    esp_gradient_file = os.path.join(DATA_FOLDER, ESP_GRADIENT_FILE)
    outfile = os.path.join(DATA_FOLDER, OUTFILE)

    # Read data
    molecules: Sequence[int, str, Atoms] = list(read_xyz(geom_file)) # List of tuples (n_atoms, comment, Atoms)
    energies: np.ndarray = load_energy_data(energy_file)
    charges: list = load_charge_data(charge_file) # Possibly ragged list
    forces: list = load_force_data(force_file) # Possibly ragged list
    if os.path.exists(esp_file) and os.path.exists(esp_gradient_file):
        print("ESP files found")
        esps, gradients = load_esp_data(esp_file, esp_gradient_file) # Possibly ragged lists
    else:
        print("ESP files not found, filling with zeros")
        esps = [np.zeros(len(molecule[2])) for molecule in molecules]
        gradients = [np.zeros((len(molecule[2]), 3)) for molecule in molecules]

    # Convert units
    energies *= H_to_eV
    # charges = [charge_array * e_to_e for charge_array in charges]
    forces = [force_matrix * H_B_to_eV_A * -1 for force_matrix in forces]
    esps = [esp_array * H_to_eV for esp_array in esps]
    electric_fields = [gradient_matrix * H_B_to_eV_A * -1 for gradient_matrix in gradients]

    write_extxyz(outfile, molecules, energies, forces, charges, esps, electric_fields)

if __name__ == "__main__":
    main()

    