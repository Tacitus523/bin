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
import seaborn as sns
import pandas as pd
from typing import Dict, List, Optional, Union

MACE_GEOMS = "geoms_fieldmace.xyz" # Should already contain reference and MACE data

PLOT_CHARGES = False
PLOT_ENERGY = True
PLOT_FORCES = True

# Keywords for extracting data
REF_ENERGY_KEY = "ref_energy"
REF_FORCES_KEY = "ref_force"
REF_CHARGES_KEY = None
PRED_ENERGY_KEY = "MACE_energy"
PRED_FORCES_KEY = "MACE_forces"
PRED_CHARGES_KEY = None


def parse_args():
    parser = argparse.ArgumentParser(description="Plot MACE data")
    parser.add_argument(
        "-g",
        "--geoms",
        type=str,
        default=MACE_GEOMS,
        help="Path to the geoms file, default: %s" % MACE_GEOMS,
    )
    parser.add_argument(
        "-s",
        "--sources",
        type=str,
        default=None,
        help="Path to the data sources file",
    )
    args = parser.parse_args()
    return args

def get_ref(
        mols,
        energy_keyword=None,
        forces_keyword=None,
        charges_keyword=None):
    ref_energy = []
    ref_forces = []
    ref_charges = []
    ref_elements = []
    
    for m in mols:
        if charges_keyword != None:
            if charges_keyword == "charge":
                ref_charges.extend(m.get_charges().flatten())
            else:
                ref_charges.extend(m.arrays[charges_keyword].flatten())
        if energy_keyword != None:
            if energy_keyword == "energy":
                ref_energy.append(m.get_potential_energy().flatten())
            else:
                ref_energy.append(m.info[energy_keyword].flatten())
        if forces_keyword != None:
            try:
                ref_forces.extend(m.info[forces_keyword].flatten())
            except:
                print(f"Warning: Forces with key '{forces_keyword}' not found for a molecule.")
        ref_elements.extend(m.get_chemical_symbols())

    ref_energy = np.array(ref_energy).flatten()
    ref_forces = np.array(ref_forces)
    ref_charges = np.array(ref_charges)
    ref_elements = np.array(ref_elements)
    return {
        "energy": ref_energy,
        "forces": ref_forces,
        "charges": ref_charges,
        "elements": ref_elements,
    }

def plot_data(ref_data, pred_data, key, x_label, y_label, filename, unit="", sources=None):
    """
    Create a scatter plot comparing reference and MACE values.
    
    Args:
        ref_data: Dictionary containing reference data
        pred_data: Dictionary containing predicted data
        key: Key to extract the specific data from dictionaries
        x_label: Label for x-axis
        y_label: Label for y-axis
        filename: Output filename for the plot
        unit: Unit for the data
        sources: Optional data sources for coloring
    """
    # Check if data exists
    if key not in ref_data or key not in pred_data:
        print(f"Warning: '{key}' not found in data dictionaries. Skipping plot.")
        return
    
    if len(ref_data[key]) == 0 or len(pred_data[key]) == 0:
        print(f"Warning: Empty data for '{key}'. Skipping plot.")
        return
    
    # Create dataframe for plotting
    df = pd.DataFrame({
        x_label: ref_data[key],
        y_label: pred_data[key],
    })
    
    # Add sources if available
    if sources is not None and len(sources) > 0:
        if len(ref_data[key]) % len(sources) == 0:
            repetitions = len(ref_data[key]) // len(sources)
            df["source"] = np.repeat(sources, repetitions)
        else:
            print(f"Warning: Number of sources doesn't match data points. Using elements instead.")
            sources = None
    
    # If no sources but we have elements (for per-atom properties)
    if sources is None and key in ["forces", "charges"] and "elements" in ref_data:
        df["source"] = ref_data["elements"]
    
    # Calculate error metrics
    rmse = np.sqrt(np.mean((df[x_label] - df[y_label]) ** 2))
    r2 = df[x_label].corr(df[y_label], method="pearson") ** 2
    
    print(f"RMSE for {key}: {rmse:.2f} {unit}")
    print(f"R² for {key}: {r2:.4f}")
    
    # Create plot with seaborn styling
    sns.set_context("talk")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot scatter with optional coloring
    sns.scatterplot(
        data=df,
        x=x_label,
        y=y_label,
        hue="source" if "source" in df.columns else None,
        palette="viridis",
        alpha=0.7,
        edgecolor=None,
    )
    
    # Plot identity line
    plt.plot(ref_data[key], ref_data[key], color="black", label="_Identity Line")
    
    # Add labels with units if provided
    plt.xlabel(f"{x_label} ({unit})" if unit else x_label)
    plt.ylabel(f"{y_label} ({unit})" if unit else y_label)
    
    # Add error metrics text box
    plt.text(
        0.70,
        0.25,
        f"RMSE: {rmse:.2f} {unit}\nR²: {r2:.4f}",
        transform=plt.gca().transAxes,
        fontsize=15,
        verticalalignment="top",
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
    )
    
    # Improve legend if available
    if "source" in df.columns:
        plt.legend(title=None, loc="upper left", fontsize="small")
        for legend_handle in ax.get_legend().legend_handles:
            legend_handle.set_alpha(1.0)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def main():
    args = parse_args()
    
    # Read molecules with basic error handling
    try:
        mace_mols = read(args.geoms, format="extxyz", index=":")
        if len(mace_mols) == 0:
            raise ValueError(f"No molecules found in file: {args.geoms}")
    except Exception as e:
        print(f"Error reading geometry file: {e}")
        return
    
    # Extract data
    ref_data = get_ref(mace_mols, REF_ENERGY_KEY, REF_FORCES_KEY, REF_CHARGES_KEY)
    MACE_data = get_ref(mace_mols, PRED_ENERGY_KEY, PRED_FORCES_KEY, PRED_CHARGES_KEY)
    
    # Read sources file if provided
    sources = None
    if hasattr(args, 'sources') and args.sources is not None:
        try:
            with open(args.sources, "r") as f:
                sources = np.array([line.strip() for line in f.readlines()])
            if len(sources) != len(mace_mols):
                print(f"Warning: Number of sources ({len(sources)}) does not match configurations ({len(mace_mols)})")
                sources = None
        except Exception as e:
            print(f"Error reading sources file: {e}")
    
    # Print data statistics
    for name, data in zip(["Ref", "FieldMACE"], [ref_data, MACE_data]):
        for key, value in data.items():
            # Skip non-numeric data
            if isinstance(value, np.ndarray) and value.dtype in (np.float32, np.float64, np.int32, np.int64) and len(value) > 0:
                print(f"{name} {key}: {value.shape} Min Max: {np.min(value): .1f} {np.max(value): .1f}")

    # Use the plot function for each data type
    if PLOT_ENERGY:
        plot_data(ref_data, MACE_data, "energy", 
                  'DFT energy', 'FieldMACE energy', "FieldMACEenergy.png", "eV", sources)

    if PLOT_CHARGES:
        plot_data(ref_data, MACE_data, "charges", 
                  'Hirshfeld charges', 'FieldMace Charges', "FieldMACEcharges.png", "e", 
                  sources if sources is not None else ref_data.get("elements"))

    if PLOT_FORCES:
        plot_data(ref_data, MACE_data, "forces", 
                  'DFT forces', 'FieldMACE forces', "FieldMACEforces.png", r"$\frac{eV}{\AA}$", 
                  sources if sources is not None else ref_data.get("elements"))

if __name__ == "__main__":
    main()