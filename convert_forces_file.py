#!/usr/bin/env python3
import argparse
import os

# Input arguments from command line.
ap = argparse.ArgumentParser(description="Handle gpu_ids and training parameters")
ap.add_argument("-g", "--geom", type=str, dest="geom", action="store", required=True, help="Path to .xyz file for mapping", metavar="geom")
ap.add_argument("-f", "--force", type=str, dest="force", action="store", required=True, help="Path to force file to convert", metavar="force")
ap.add_argument("-o", "--out", type=str, default="forces_conv.xyz", dest="output", action="store", required=False, help="Path to output file", metavar="output")
args = vars(ap.parse_args())
print("Input of argparse:", args)

# Read the input files.
geom = open(args["geom"], "r")
force = open(args["force"], "r")

with open(args["output"], "w") as out:
    # Read the initial number of atoms.
    line = geom.readline()
    while line:
        n_atoms = int(line)
        out.write(str(n_atoms) + "\n")
        # Write the comment line.
        out.write(geom.readline())
        for i in range(n_atoms):
            # Read the atom and its coordinates.
            atom, x, y, z = geom.readline().split()
            # Read the forces.
            fx, fy, fz = force.readline().split()

            out.write(f"{atom} {fx} {fy} {fz}\n")
        # Read the next number of atoms.
        line = geom.readline()


geom.close()
force.close()