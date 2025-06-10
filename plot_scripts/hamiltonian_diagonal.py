#!/usr/bin/env python3
import argparse
import numpy as np
import os
import sys
from typing import Optional, List, Dict

from ase.io import read
from ase import Atoms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mace_scripts.MacePlot import extract_data, plot_histogram

# Keywords for extracting data
REF_ENERGY_KEY: Optional[str] = "ref_energy"
REF_FORCES_KEY: Optional[str] = "ref_force"
REF_CHARGES_KEY: Optional[str] = "ref_charge"
REF_DMA_KEY: Optional[str] = None

HAMILTONIAN_DIAGONAL_UNITS = "eV"
PLOT_FILENAME = "hamiltonian_diagonal_histogram.png"

hartree_to_eV = 27.211386245988

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Hamiltonian diagonal elements.")
    parser.add_argument("-g", "--geoms", type=str, required=True,
                        help="Path to the geometry files.")
    parser.add_argument("-d", "--diagonal", type=str, required=True,
                        help="Path to the Hamiltonian diagonal elements file.")
    args = parser.parse_args()
    return args

def load_hamiltonian_data(hamiltonian_file: str) -> np.ndarray:
    """
    Load Hamiltonian diagonal elements in atomic units from a file.
    
    Args:
        hamiltonian_file (str): Path to the Hamiltonian file.
    
    Returns:
        np.ndarray: Array of Hamiltonian diagonal elements.
    """
    return np.loadtxt(hamiltonian_file)

def main():
    args = parse_args()
    
    mace_mols: List[Atoms] = read(args.geoms, format="extxyz", index=":")
    hamiltonian_diagonals: np.ndarray = load_hamiltonian_data(args.diagonal)
    n_hamiltonian = hamiltonian_diagonals.shape[0]
    mace_mols = mace_mols[:n_hamiltonian]
    hamiltonian_diagonals = hamiltonian_diagonals.flatten()
    hamiltonian_diagonals = hamiltonian_diagonals * hartree_to_eV  # Convert from Hartree to eV

    ref_data: Dict[str, np.ndarray] = extract_data(
        mace_mols, REF_ENERGY_KEY, REF_FORCES_KEY, REF_CHARGES_KEY, REF_DMA_KEY
    )
    assert ref_data["elements"].shape == hamiltonian_diagonals.shape, \
        f"Mismatch between number of Hamiltonian diagonal elements and number of geometries: {hamiltonian_diagonals.shape} vs {ref_data['elements'].shape}"
    ref_data["hamiltonian_diagonal"] = hamiltonian_diagonals
    
    plot_histogram(ref_data, keys=["hamiltonian_diagonal"], units=[HAMILTONIAN_DIAGONAL_UNITS], filename=PLOT_FILENAME)


if __name__ == "__main__":
    main()