#!/usr/bin/env python3
import json
import numpy as np
import argparse
import os

#sym_func_json = "/home/leichinger/git/phosML/dftb-nn/nn_Behler/saved_models/keks/symfunc_params.json"

def parse_args():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sym_func_json", type=str, default="symfunc_params.json", help="Path to the symmetry function json file")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    sym_func_json = args.sym_func_json

    print(f"Reading symmetry function parameters from {sym_func_json}")

    angstrom_to_bohr = 1.8897259886 # Copied from consts_params.py

    with open(sym_func_json, "r") as f:
        sym_func = json.load(f)

    radial_parameters = np.array(sym_func["radial_parameters"])
    angular_parameters = np.array(sym_func["angular_parameters"])

    radial_cutoff = radial_parameters[:, 2]
    assert np.all(radial_cutoff == radial_cutoff[0]), "radial cutoffs are not the same"
    radial_cutoff = radial_cutoff[0]/angstrom_to_bohr
    radial_parameters = radial_parameters[:, :2]
    radial_parameters[:, 0] = radial_parameters[:, 0]/angstrom_to_bohr
    radial_parameters[:, 1] = radial_parameters[:, 1]*(angstrom_to_bohr**2)

    angular_cutoff = angular_parameters[:, 3]
    assert np.all(angular_cutoff == angular_cutoff[0]), "angular cutoffs are not the same"
    angular_cutoff = angular_cutoff[0]/angstrom_to_bohr
    angular_parameters = angular_parameters[:, :3]
    angular_parameters[:, 0] = angular_parameters[:, 0]*(angstrom_to_bohr**2)

    print(f"RadialCutoff = {radial_cutoff:.2f}")
    print("RadialParameters {")
    for i, (eta, rs) in enumerate(radial_parameters):
        print(f"   {eta:.4f}, {rs:.4f} ")
    print("}")

    print(f"AngularCutoff = {angular_cutoff:.2f}")
    print("AngularParameters {")
    for i, (eta, zeta, lamb) in enumerate(angular_parameters):
        print(f"  {eta:.4f}, {zeta:.4f}, {lamb:.4f} ")
    print("}")


if __name__ == "__main__":
    main()

