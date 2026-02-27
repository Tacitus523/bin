#!/usr/bin/env python
import argparse
import h5py
import numpy as np

H_TO_EV = 27.211386245988
BOHR_TO_A = 0.52917721067
H_B_TO_EV_A = H_TO_EV / BOHR_TO_A

parser = argparse.ArgumentParser(description="Check HDF5 training data.")
parser.add_argument("-f", "--file", type=str, default="disulfide.hdf5", help="Path to the HDF5 file.")
args = parser.parse_args()

with h5py.File(args.file, "r") as f:
    all_energies = []
    all_forces = []
    n_datapoints = 0

    for group_name in f.keys():
        group = f[group_name]
        if "qm_energies" in group:
            energies = group["qm_energies"][:] * H_TO_EV
            all_energies.append(energies)
            n_datapoints += len(energies)
        else:
            raise ValueError(f"'qm_energies' not found in group. Available: {list(group.keys())}")
        if "qm_forces" in group:
            forces = group["qm_forces"][:] * H_B_TO_EV_A
            all_forces.append(forces.reshape(-1))
        elif "qm_gradients" in group:
            forces = -group["qm_gradients"][:] * H_B_TO_EV_A
            all_forces.append(forces.reshape(-1))

    all_energies = np.concatenate(all_energies)
    all_forces = np.concatenate(all_forces)

    print(f"Number of datapoints: {n_datapoints}")
    print(f"Energy range: {all_energies.min():.4f} to {all_energies.max():.4f} eV")
    print(f"Force components range: {all_forces.min():.4f} to {all_forces.max():.4f} eV/Å")
