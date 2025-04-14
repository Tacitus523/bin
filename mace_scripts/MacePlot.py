#!/usr/bin/env python
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --job-name=plot
#SBATCH --output=plot.out
#SBATCH --error=plot.out

import argparse
from ase.io import read,write
import numpy as np
import matplotlib.pyplot as plt

MACE_GEOMS = "geoms_mace.extxyz" # Should already contain reference and MACE data

PLOT_CHARGES = True 
PLOT_ENERGY = True
PLOT_FORCES = True
PLOT_DMA = False

# Keywords for extracting data
REF_ENERGY_KEY = "ref_energy"
REF_FORCES_KEY = "ref_force"
REF_CHARGES_KEY = "ref_charge"
REF_DMA_KEY = None
PRED_ENERGY_KEY = "MACE_energy"
PRED_FORCES_KEY = "MACE_forces"
PRED_CHARGES_KEY = "MACE_charges"
PRED_DMA_KEY = None

def parse_args():
    parser = argparse.ArgumentParser(description="Plot MACE data")
    parser.add_argument(
        "-g",
        "--geoms",
        type=str,
        default=MACE_GEOMS,
        help="Path to the geoms file, default: %s" % MACE_GEOMS,
    )
    args = parser.parse_args()
    return args

def get_ref(
        mols,
        energy_keyword=None,
        forces_keyword=None,
        charges_keyword=None,
        DMA_keyword=None):
    ref_energy = []
    ref_forces = []
    ref_charges = []
    ref_DMA = []
    for m in mols:
        if charges_keyword != None:
            if charges_keyword == "charge":
                ref_charges.extend(m.get_charges())
            else:
                ref_charges.extend(m.arrays[charges_keyword])
        if energy_keyword != None:
            if energy_keyword == "energy":
                ref_energy.append(m.get_potential_energy())
            else:
                ref_energy.append(m.info[energy_keyword])
        if forces_keyword != None:
            if forces_keyword == "forces":
                ref_forces.extend(m.get_forces().flatten())
            else:
                ref_forces.extend(m.arrays[forces_keyword].flatten())
        if DMA_keyword != None:
            AIMS_atom_multipoles = m.arrays[DMA_keyword]
            ref_DMA.extend(AIMS_atom_multipoles[:,0])
    ref_energy = np.array(ref_energy)
    ref_forces = np.array(ref_forces)
    ref_charges = np.array(ref_charges)
    ref_DMA = np.array(ref_DMA)
    return {
        "energy": ref_energy,
        "forces":ref_forces,
        "charges":ref_charges,
        "DMA":ref_DMA
    }

def plot_data(ref_data, pred_data, key, x_label, y_label, filename):
    """
    Create a scatter plot comparing reference and MACE values.
    
    Args:
        ref_data: Dictionary containing reference data
        pred_data: Dictionary containing predicted data
        key: Key to extract the specific data from dictionaries
        x_label: Label for x-axis
        y_label: Label for y-axis
        filename: Output filename for the plot
    """
    plt.figure()
    plt.scatter(ref_data[key], pred_data[key], c='blue', alpha=0.5, label='Data Points')  # Scatter plot
    plt.plot(ref_data[key], ref_data[key], color="black", label='Identity Line')  # Identity line
    plt.xlabel(x_label)  # X-axis Label
    plt.ylabel(y_label)  # Y-axis Label
    plt.tight_layout()  # Tight layout for nicer appearance
    plt.savefig(filename, dpi=300)
    plt.close()



def main():
    args = parse_args()
    mace_mols = read(args.geoms, format="extxyz", index=":")
    ref_data = get_ref(mace_mols,"ref_energy","ref_force","ref_charge", None)
    MACE_data = get_ref(mace_mols, "MACE_energy", "MACE_forces", "MACE_charges", None)

    # Use the plot function for each data type
    if PLOT_ENERGY:
        plot_data(ref_data, MACE_data, "energy", 
                  'DFT energy', 'MACE energy', "MACEenergy.png")

    if PLOT_DMA:
        plot_data(ref_data, MACE_data, "DMA", 
                  'DMA ref', 'Mace DMA', "MACEdma.png")
        
    if PLOT_CHARGES:
        plot_data(ref_data, MACE_data, "charges", 
                  'Hirshfeld charges', 'Mace Charges', "MACEcharges.png")

    if PLOT_FORCES:
        plot_data(ref_data, MACE_data, "forces", 
                  'dft forces', 'mace forces', "MACEforces.png")

if __name__ == "__main__":
    main()