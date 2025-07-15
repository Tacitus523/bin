#!/usr/bin/env python3
# Gets a trajectory and a topology, checks the last frame for explosion indefinitely
# Breaks the loop if explosion is detected
import argparse
import os
import time
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from ase.data import atomic_numbers
import MDAnalysis as mda
import numpy as np


MAX_WAIT_TIME_INITIALIZATION = 120 # seconds to wait for initialization
SLEEP_TIME = 30 # seconds to sleep between checks
EXPLOSION_THRESHOLD = 4.0 # Angstroms, threshold for explosion detection
DEFAULT_BASENAME = "run"
DEFAULT_TRAJECTORY = "{}.xtc"
DEFAULT_TOPOLOGY = "{}.tpr"
REPLACEMENT_TRAJECTORY = "{}.trr"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Observe trajectory for explosion.")
    parser.add_argument("--basename", type=str, required=False, default=None, 
                        help=f"Base name for the trajectory and topology files (default: {DEFAULT_BASENAME}).")
    parser.add_argument("--trajectory", type=str, required=False, default=None,
                        help=f"Path to the trajectory file (default: {DEFAULT_TRAJECTORY.format(DEFAULT_BASENAME)}).")
    parser.add_argument("--topology", type=str, required=False, default=None,
                        help=f"Path to the topology file (default: {DEFAULT_TOPOLOGY.format(DEFAULT_BASENAME)}).")
    parser.add_argument("-o", "--once", action="store_true",
                        help="Run the observation once and exit, instead of running indefinitely.")
    args = parser.parse_args()
    validate_args(args)
    return args

def validate_args(args: argparse.Namespace) -> None:
    if args.basename is None:
        args.basename = DEFAULT_BASENAME
    if args.trajectory is None:
        args.trajectory = DEFAULT_TRAJECTORY.format(args.basename)
        args.replacement_trajectory = REPLACEMENT_TRAJECTORY.format(args.basename)
    else:
        args.replacement_trajectory = args.trajectory.replace(".xtc", ".trr")
    if args.topology is None:
        args.topology = DEFAULT_TOPOLOGY.format(args.basename)
    
    elapsed_time = 0
    time_increment = 10  # seconds
    while elapsed_time < MAX_WAIT_TIME_INITIALIZATION:
        if os.path.exists(args.topology) and os.path.exists(args.trajectory):
            break
        if os.path.exists(args.topology) and os.path.exists(args.replacement_trajectory):
            args.trajectory = args.replacement_trajectory
            break
        time.sleep(time_increment)
        elapsed_time += time_increment
    if not os.path.exists(args.trajectory):
        raise FileNotFoundError(f"Trajectory file {args.trajectory} does not exist.")
    if not os.path.exists(args.topology):
        raise FileNotFoundError(f"Topology file {args.topology} does not exist.")
    
def observe_trajectory(trajectory: str, topology: str) -> bool:
    universe: mda.Universe = mda.Universe(topology, trajectory)
    all_atoms: mda.AtomGroup = universe.select_atoms("all")
    all_atoms.wrap()  # Ensure atoms are wrapped in the box
    unique_edge_indices, elements_bonds, atomic_numbers_bonds = get_atomic_numbers_and_elements(all_atoms)
    last_frame: mda.coordinates.base.Timestep = universe.trajectory[-1]
    distance_matrix = mda.lib.distances.distance_array(all_atoms.positions, all_atoms.positions, box=universe.dimensions)
    bond_distances = distance_matrix[unique_edge_indices[:, 0], unique_edge_indices[:, 1]]
    explosion_detected = np.any(bond_distances > EXPLOSION_THRESHOLD)
    return explosion_detected

def get_atomic_numbers_and_elements(atoms: mda.AtomGroup) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the atomic numbers and elements of the atoms in the bonds.

    Args:
        atoms (mda.AtomGroup): The atoms in the system.

    Returns:
        Tuple[inp.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing the following:
            - unique_edge_indices (np.ndarray): An array of unique edge indices.
            - elements_bonds (np.ndarray): An array of elements of the atoms in the bonds. Guesses the element symbol from the atom name.
            - atomic_numbers_bonds (np.ndarray): An array of atomic numbers of the atoms in the bonds. Depends on the element symbol guess.
    """
    unique_edge_indices = []
    elements_bonds = []
    atomic_numbers_bonds = []
    for atom_a in atoms:
        for atom_b in atom_a.bonded_atoms:
            if atom_a.index < atom_b.index:
                unique_index = [atom_a.index, atom_b.index]
                elements_bond = [atom_a.element, atom_b.element]
                atomic_numbers_bond = [atomic_numbers.get(element, 0) for element in elements_bond]
                unique_edge_indices.append(unique_index)
                elements_bonds.append(elements_bond)
                atomic_numbers_bonds.append(atomic_numbers_bond)

    unique_edge_indices = np.array(unique_edge_indices) # shape: (n_bonds, 2)
    elements_bonds = np.array(elements_bonds) # shape: (n_bonds, 2)
    atomic_numbers_bonds = np.array(atomic_numbers_bonds) # shape: (n_bonds, 2)

    return unique_edge_indices, elements_bonds, atomic_numbers_bonds

def main() -> None:
    args = parse_args()
    while True:
        explosion_detected = observe_trajectory(args.trajectory, args.topology)
        if explosion_detected:
            print("Explosion detected in the last frame of the trajectory.")
            break
        elif args.once:
            print("No explosion detected in the last frame of the trajectory. Exiting.")
            break
        else:
            time.sleep(SLEEP_TIME)

if __name__ == "__main__":
    main()

