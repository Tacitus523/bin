#!/usr/bin/env python3
# Script for getting the electronegativity out of square hamiltonian diagonals in the orbital basis
# Hardcoded element order for now, just keeps one of the p-oritals for heavy elements and the s-orbital for Hydrogen
import argparse
import numpy as np

ELEMENTAL_ORDER = ["C","H","H","H","C","O","N","H","C","H","C","H","H","H","C","O","N","H","C","H","H","H"]
SAVE_PATH = "hamiltonian_electronegativities.txt"

target_orbitals ={
    "H": 0,
    "C": 1,
    "N": 1,
    "O": 1,
}

max_orbitals = {
    "H": 1,
    "C": 4,
    "N": 4,
    "O": 4,
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract DFTB Hamiltonian diagonal elements.")
    parser.add_argument("filename", type=str, help="Path to the file containing Hamiltonian diagonal elements.")
    return parser.parse_args()

def load_hamiltonian_diagonals(filename: str) -> np.ndarray:
    """
    Load the Hamiltonian diagonal elements from a file.
    The file should contain one diagonal element per line.
    """
    return np.loadtxt(filename)


def get_target_orbitals(hamiltonian_diagonals: np.ndarray) -> np.ndarray:
    """
    Extract the target orbitals from the Hamiltonian diagonals.
    """
    cumulative_indices = np.cumsum([0]+[max_orbitals[element] for element in ELEMENTAL_ORDER][:-1]) # End indices of each element's orbitals
    #cumulative_indices = cumulative_indices - 1  # Convert to zero-based indexing
    target_indices = np.array([target_orbitals[element] for element in ELEMENTAL_ORDER])  # Get the target orbital index for each element
    target_indices = target_indices + cumulative_indices  # Add the cumulative indices to get the correct indices in the flattened array
    print(f"Cumulative indices for target orbitals: {target_indices}")
    return hamiltonian_diagonals[:,target_indices]

def main() -> None:
    args = parse_args()
    hamiltonian_diagonals = load_hamiltonian_diagonals(args.filename)
    
    if hamiltonian_diagonals.ndim != 2:
        raise ValueError("Input file must contain a 2D array with the correct number of columns.")

    target_orbitals = get_target_orbitals(hamiltonian_diagonals)
    assert target_orbitals.shape[1] == len(ELEMENTAL_ORDER), "Number of target orbitals does not match the number of elements in ELEMENTAL_ORDER."

    np.savetxt(SAVE_PATH, target_orbitals, fmt='%.6f')

if __name__ == "__main__":
    main()