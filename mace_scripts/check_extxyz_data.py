#!/usr/bin/env python
import argparse
from ase.io import read
import numpy as np

parser = argparse.ArgumentParser(description="Check extxyz training data.")
parser.add_argument("-f", "--file", type=str, default="geoms.extxyz", help="Path to the extxyz file.")
args = parser.parse_args()

mols = read(args.file, index=":")
print(f"Number of datapoints: {len(mols)}")

energies = np.array([m.info["ref_energy"] for m in mols])
print(f"Energy range: {energies.min():.4f} to {energies.max():.4f} eV")

forces = np.concatenate([m.arrays["ref_force"].flatten() for m in mols])
print(f"Force components range: {forces.min():.4f} to {forces.max():.4f} eV/Å")