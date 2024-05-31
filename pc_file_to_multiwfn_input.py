import argparse

import numpy as np


nm_to_bohr = 18.897259885789
A_to_bohr = 1.8897259886

def main():
    ap = argparse.ArgumentParser(description="Conversion of an Orca pointcharge file to a molden readable format")
    ap.add_argument("-f", "--file", type=str, dest="pointcharge_file", action="store", required=True, help="Orca pointcharge file", metavar="pointcharge_file")
    ap.add_argument("-o", "--out", default="temp.pc", type=str, dest="out_file", action="store", required=False, help="Output file, default 'temp.pc'", metavar="out_file")
    args = ap.parse_args()
    pointcharge_file = args.pointcharge_file
    out_file = args.out_file

    n_mm_atoms = np.loadtxt(pointcharge_file, max_rows=1, dtype=int)
    mm_coords = np.loadtxt(pointcharge_file, skiprows=1, usecols=(1,2,3)) *A_to_bohr
    np.savetxt(out_file, mm_coords, header=str(n_mm_atoms), comments='')

if __name__=="__main__":
    main()
