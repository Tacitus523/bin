#!/home/lpetersen/miniconda3/envs/kgcnn_new/bin/python
import argparse

import numpy as np

N_SUBSAMPLE = np.inf

nm_to_bohr = 18.897259885789
A_to_bohr = 1.8897259886

def main():
    ap = argparse.ArgumentParser(description="Conversion of an Orca pointcharge file to a molden readable format")
    ap.add_argument("-f", "--file", type=str, dest="pointcharge_file", action="store", required=True, help="Orca pointcharge file", metavar="pointcharge_file")
    ap.add_argument("-n", "--n_subsample", default=N_SUBSAMPLE, type=int, dest="n_subsample", action="store", required=False, help=f"Maximum number of MM atoms to include, default {N_SUBSAMPLE}", metavar="n_subsample")
    ap.add_argument("-o", "--out", default="temp.pc", type=str, dest="out_file", action="store", required=False, help="Output file, default 'temp.pc'", metavar="out_file")
    args = ap.parse_args()
    pointcharge_file = args.pointcharge_file
    out_file = args.out_file

    # Read pointcharge file
    n_mm_atoms = np.loadtxt(pointcharge_file, max_rows=1, dtype=int)
    mm_charges = np.loadtxt(pointcharge_file, skiprows=1, usecols=(0,))
    mm_coords = np.loadtxt(pointcharge_file, skiprows=1, usecols=(1,2,3)) *A_to_bohr

    n_mm_atoms_subsample = min(n_mm_atoms, args.n_subsample)
    if n_mm_atoms > args.n_subsample:
        # Sort by distance from center
        mm_center = np.mean(mm_coords, axis=0)
        dist_from_center = np.linalg.norm(mm_coords - mm_center, axis=1)
        sorted_indices = np.argsort(dist_from_center)
        mm_charges = mm_charges[sorted_indices[:n_mm_atoms_subsample]]
        mm_coords = mm_coords[sorted_indices[:n_mm_atoms_subsample]]

    # # Subsample if too many atoms
    # n_mm_atoms_subsample = min(n_mm_atoms, args.n_subsample)
    # random_indices = np.random.choice(n_mm_atoms, size=n_mm_atoms_subsample, replace=False)
    # random_indices.sort()
    # mm_charges = mm_charges[random_indices]
    # mm_coords = mm_coords[random_indices]
    mm_data =  np.concatenate([mm_coords, mm_charges[:, np.newaxis]], axis=1) # Only need coords in Bohr(first three columns), but charges are useful later

    # Write output file
    np.savetxt(out_file, mm_data, header=str(n_mm_atoms_subsample), comments='', fmt='%.6f %.6f %.6f %.4f', delimiter=' ')

if __name__=="__main__":
    main()
