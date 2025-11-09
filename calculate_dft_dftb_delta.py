#!/usr/bin/env python
"""
Calculate the delta between DFT and DFTB energies and forces from an .extxyz file.
Delta energy = E_DFT - E_DFTB
Delta forces = F_DFT - F_DFTB
Units: eV for energies, eV/Å for forces
"""
import yaml
from typing import List, Optional, Dict

import argparse
import numpy as np
from ase import Atoms
from ase.io import read, write
from scipy.spatial.distance import pdist

ENERGY_KEY = "ref_energy"
FORCES_KEY = "ref_force"
REQUIERED_KEYS = [] # "Lattice", "pbc" are both used by ase internally, access them directly from atoms objects via atoms.get_cell() and atoms.get_pbc()
OPTIONAL_KEYS = ["data_source", "esp", "esp_gradient", "total_charge"]

DELTA_ATOMIZATION_ENERGY_KEY = "delta_atomization_energy"
DEFAULT_OUTPUT = "geoms_delta.extxyz"
E0_FILE_OUTPUT = "delta_E0s.yaml"

# /data/lpetersen/atomic_energies/B3LYP_aug-cc-pVTZ
DFT_E0s={
    1: -13.575035506869515, 
    6: -1029.6173622986487, 
    7: -1485.1410643783852, 
    8: -2042.617308911902,
    16: -10832.265333248919
}

# /data/lpetersen/atomic_energies/DFTB_3OB
DFTB_E0s={
   1: -7.609986074389834,
   6: -39.29249996225988,
   7: -60.326270220805434,
   8: -85.49729667072424,
   16: -63.22114702401557
}

# # /data/lpetersen/atomic_energies/DFTB_3OB_mod_S-S
# DFTB_E0s={
#    1: -7.609986074389834,
#    6: -39.29249996225988,
#    7: -60.326270220805434,
#    8: -85.49729667072424,
#    16: -63.22114702401557
# }

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate DFT - DFTB energy and force deltas from .extxyz file.")
    parser.add_argument("input_files", type=str, nargs=2, help="Paths to .extxyz files: [0] DFT, [1] DFTB")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Output .extxyz file for delta geometries")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance for geometry comparison (default: 1e-5)")
    parser.add_argument("--atol", type=float, default=1e-8, help="Absolute tolerance for geometry comparison (default: 1e-8)")
    args = parser.parse_args()
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    return args

def assert_geometry_equivalence(
        atoms_list_1: List[Atoms], 
        atoms_list_2: List[Atoms], 
        rtol: float = 1e-5, 
        atol: float = 1e-8
    ) -> None:
    """
    Assert that two lists of atoms have equivalent geometries by comparing distance matrices.
    
    Args:
        atoms_list_1: List of ASE Atoms objects from first file
        atoms_list_2: List of ASE Atoms objects from second file
        rtol: Relative tolerance for numpy.allclose
        atol: Absolute tolerance for numpy.allclose
    
    Raises:
        AssertionError: If geometries are not equivalent
    """
    if len(atoms_list_1) != len(atoms_list_2):
        raise AssertionError(f"Number of geometries differ: {len(atoms_list_1)} vs {len(atoms_list_2)}")
    
    for geom_idx in range(len(atoms_list_1)):
        atoms_1 = atoms_list_1[geom_idx]
        atoms_2 = atoms_list_2[geom_idx]
        
        if len(atoms_1) != len(atoms_2):
            raise AssertionError(f"Geometry {geom_idx}: Number of atoms differ: {len(atoms_1)} vs {len(atoms_2)}")
        
        # Compare distance matrices
        dist_matrix_1 = pdist(atoms_1.get_positions())
        dist_matrix_2 = pdist(atoms_2.get_positions())
        
        if not np.allclose(dist_matrix_1, dist_matrix_2, rtol=rtol, atol=atol):
            diff = np.abs(dist_matrix_1 - dist_matrix_2)
            max_diff = np.max(diff)
            max_idx = np.argmax(diff)
            
            raise AssertionError(
                f"Geometry {geom_idx}: Distance matrices differ.\n"
                f"Maximum difference: {max_diff:.2e} at pairwise distance {max_idx}\n"
                f"DFT value: {dist_matrix_1[max_idx]:.6f}\n"
                f"DFTB value: {dist_matrix_2[max_idx]:.6f}"
            )

def create_delta_file(
        dft_atoms_list: List[Atoms], 
        dftb_atoms_list: List[Atoms], 
        delta_E0s: Dict[int, float], 
        args: argparse.Namespace
    ) -> None:
    """
    Create a new .extxyz file containing the delta energies and forces between DFT and DFTB.
    
    Args:
        dft_atoms_list: List of ASE Atoms objects from DFT file
        dftb_atoms_list: List of ASE Atoms objects from DFTB file
        output_file: Path to output .extxyz file
    """
    delta_atoms_list = []
    
    for i, (dft_atoms, dftb_atoms) in enumerate(zip(dft_atoms_list, dftb_atoms_list)):
        for key in REQUIERED_KEYS + [ENERGY_KEY, FORCES_KEY]:
            if key not in dft_atoms.info and key not in dft_atoms.arrays:
                raise KeyError(f"Required key '{key}' missing in geometry {i} of DFT data.")
            if key not in dftb_atoms.info and key not in dftb_atoms.arrays:
                raise KeyError(f"Required key '{key}' missing in geometry {i} of DFT or DFTB data.")

        # Create new atoms object with DFT geometry (since geometries are equivalent)
        positions = dft_atoms.get_positions()
        atomic_numbers = dft_atoms.get_atomic_numbers()
        lattice = dft_atoms.get_cell()
        pbc = dft_atoms.get_pbc()
        delta_atoms = Atoms(positions=positions, numbers=atomic_numbers, cell=lattice, pbc=pbc)
        
        # Calculate energy delta (DFT - DFTB)
        dft_energy = dft_atoms.info[ENERGY_KEY]
        dftb_energy = dftb_atoms.info[ENERGY_KEY]
        delta_energy = dft_energy - dftb_energy
        delta_atoms.info[ENERGY_KEY] = delta_energy

        # Calculate atomization energy delta
        atomic_energy_sum = np.sum([
            delta_E0s[Z] for Z in atomic_numbers
        ])
        delta_atomization_energy = delta_energy - atomic_energy_sum
        delta_atoms.info[DELTA_ATOMIZATION_ENERGY_KEY] = delta_atomization_energy
        
        # Calculate force deltas (DFT - DFTB)
        dft_forces = dft_atoms.arrays[FORCES_KEY]
        dftb_forces = dftb_atoms.arrays[FORCES_KEY]
        delta_forces = dft_forces - dftb_forces
        delta_atoms.arrays[FORCES_KEY] = delta_forces

        # Copy required metadata
        for key in REQUIERED_KEYS:
            delta_atoms.info[key] = dft_atoms.info[key]

        # Copy optional metadata
        for key in OPTIONAL_KEYS:
            if key in dft_atoms.info:
                delta_atoms.info[key] = dft_atoms.info[key]
        
        delta_atoms_list.append(delta_atoms)
    
    # Write delta geometries
    print(f"Writing {len(delta_atoms_list)} delta geometries to {args.output}")
    write(args.output, delta_atoms_list)
    
    # Print statistics
    energies = [atoms.info[DELTA_ATOMIZATION_ENERGY_KEY] for atoms in delta_atoms_list]
    forces_flat = np.concatenate([atoms.arrays[FORCES_KEY].flatten() for atoms in delta_atoms_list])
    
    print(f"\nDelta Atomization Energy Statistics:")
    print(f"  Mean: {np.mean(energies):.6f} eV")
    print(f"  Std:  {np.std(energies):.6f} eV")
    print(f"  Min:  {np.min(energies):.6f} eV")
    print(f"  Max:  {np.max(energies):.6f} eV")
    
    print(f"\nDelta Force Statistics:")
    print(f"  RMS:  {np.sqrt(np.mean(forces_flat**2)):.6f} eV/Å")
    print(f"  Mean: {np.mean(forces_flat):.6f} eV/Å")
    print(f"  Std:  {np.std(forces_flat):.6f} eV/Å")

def calculate_delta_E0s(dft_E0s: Dict[int, float], dftb_E0s: Dict[int, float]) -> Dict[int, float]:
    assert dft_E0s.keys() == dftb_E0s.keys(), "Element keys in DFT and DFTB E0s do not match."
    delta_E0s = {}
    for element in dft_E0s.keys():
        delta_E0 = dft_E0s[element] - dftb_E0s[element]
        delta_E0s[element] = delta_E0
    
    with open(E0_FILE_OUTPUT, "w") as f:
        yaml.dump(delta_E0s, f)
    print(f"Delta E0s (DFT - DFTB) saved to {E0_FILE_OUTPUT}")

    return delta_E0s

def main() -> None:
    args = parse_arguments()
    dft_file, dftb_file = args.input_files
    
    print(f"Loading DFT geometries from {dft_file}")
    dft_atoms_list = read(dft_file, index=":")
    print(f"Loading DFTB geometries from {dftb_file}")
    dftb_atoms_list = read(dftb_file, index=":")
    
    print(f"DFT: {len(dft_atoms_list)} geometries, {len(dft_atoms_list[0])} atoms each")
    print(f"DFTB: {len(dftb_atoms_list)} geometries, {len(dftb_atoms_list[0])} atoms each")
    
    # Assert geometry equivalence
    print("Asserting geometry equivalence...")
    assert_geometry_equivalence(dft_atoms_list, dftb_atoms_list, rtol=args.rtol, atol=args.atol)
    print("✓ Geometries are equivalent")

    # Calculate delta E0s
    print("\nCalculating delta E0s (DFT - DFTB)...")
    delta_E0s = calculate_delta_E0s(DFT_E0s, DFTB_E0s)
    
    # Calculate deltas
    print("Calculating energy and force deltas...")
    create_delta_file(dft_atoms_list, dftb_atoms_list, delta_E0s, args)

    
if __name__ == "__main__":
    main()

