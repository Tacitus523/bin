#!/usr/bin/python3
import argparse
import numpy as np

def main():
    ap = argparse.ArgumentParser(description="Calculates the energy difference to a file with optimized fragment molecules")
    ap.add_argument("-e", "--eng", type=str, dest="energy_file", action="store", required=True, help="File with unprocessed energy values", metavar="energy_file")
    ap.add_argument("-r", "--ref", type=str, dest="reference_file", action="store", required=True, help="File with reference energies", metavar="reference_file")
    ap.add_argument("-o", "--out", default="energy_diff.txt", type=str, dest="out_file", action="store", required=False, help="Output file, default 'energy_diff.txt'", metavar="out_file")
    args = ap.parse_args()
    energy_file = args.energy_file
    reference_file = args.reference_file
    out_file = args.out_file

    energies = np.loadtxt(energy_file)
    reference_energy = np.loadtxt(reference_file)
    reference_energy = np.sum(reference_energy)
    energy_diff = energies - reference_energy
    np.savetxt(out_file, energy_diff)

main()