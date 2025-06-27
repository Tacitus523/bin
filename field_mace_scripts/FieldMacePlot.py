#!/usr/bin/env python
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --job-name=plot
#SBATCH --output=plot.out
#SBATCH --error=plot.out

import argparse
from typing import Dict, List, Optional, Union
from ase.atoms import Atoms
from ase.io import read
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

MACE_GEOMS: str = "geoms_fieldmace.xyz" # Should already contain reference and MACE data
DATA_SOURCES_FILE: Optional[str] = None  # File containing the data source of each entry

PLOT_CHARGES: bool = False
PLOT_ENERGY: bool = True
PLOT_FORCES: bool = True

# Keywords for extracting data
REF_ENERGY_KEY: Optional[str] = "ref_energy"
REF_FORCES_KEY: Optional[str] = "ref_force"
REF_CHARGES_KEY: Optional[str] = None
PRED_ENERGY_KEY: Optional[str] = "MACE_energy"
PRED_FORCES_KEY: Optional[str] = "MACE_forces"
PRED_CHARGES_KEY: Optional[str] = None

# Units for plotting
ENERGY_UNIT: str = "eV"
FORCES_UNIT: str = r"$\frac{eV}{\AA}$"
CHARGES_UNIT: str = "e"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plotting script for FieldMace data")
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
        default=DATA_SOURCES_FILE,
        help="Path to the data sources file, default: %s" % DATA_SOURCES_FILE,
    )
    args = parser.parse_args()
    return args

def extract_data(
    mols: List[Atoms],
    energy_keyword: Optional[str] = None,
    forces_keyword: Optional[str] = None,
    charges_keyword: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    ref_energy: List[float] = []
    ref_forces: List[float] = []
    ref_charges: List[float] = []
    ref_elements: List[str] = []
    for m in mols:
        if charges_keyword is not None:
            if charges_keyword == "charge":
                ref_charges.extend(m.get_charges().flatten())
            else:
                ref_charges.extend(m.arrays[charges_keyword].flatten())
        if energy_keyword is not None:
            if energy_keyword == "energy":
                ref_energy.append(m.get_potential_energy().flatten())
            else:
                ref_energy.append(m.info[energy_keyword].flatten())
        if forces_keyword is not None:
            if forces_keyword == "forces":
                ref_forces.extend(m.get_forces().flatten())
            else:
                ref_forces.extend(m.arrays[forces_keyword].flatten())
        ref_elements.extend(m.get_chemical_symbols())

    result = {}
    result["energy"] = np.array(ref_energy).flatten()  # Energy in eV
    result["forces"] = np.array(ref_forces)  # Forces in eV/Å
    if len(ref_charges) > 0:
        result["charges"] = np.array(ref_charges)
    result["elements"] = np.array(ref_elements)
    return result

def plot_data(
    ref_data: Dict[str, np.ndarray],
    pred_data: Dict[str, np.ndarray],
    key: str,
    sources: Optional[np.ndarray],
    x_label: str,
    y_label: str,
    unit: str,
    filename: str,
) -> None:
    """
    Create a scatter plot comparing reference and MACE values.
    
    Args:
        ref_data: Dictionary containing reference data
        pred_data: Dictionary containing predicted data
        key: Key to extract the specific data from dictionaries
        sources: Optional data sources for coloring
        x_label: Label for x-axis
        y_label: Label for y-axis
        unit: Unit for the data
        filename: Output filename for the plot
    """
    df: pd.DataFrame = create_dataframe(ref_data, pred_data, key, sources, x_label, y_label)
    rmse: float = np.sqrt(np.mean((df[x_label] - df[y_label]) ** 2))
    r2: float = df[x_label].corr(df[y_label], method="pearson") ** 2

    print(f"RMSE for {key}: {rmse:.2f} {unit}")
    print(f"R² for {key}: {r2:.4f}")

    sns.set_context("talk")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x=x_label,
        y=y_label,
        hue="source" if sources is not None else None,
        palette="tab10",
        alpha=0.6,
        edgecolor=None,
        s=20,
    )
    plt.plot(ref_data[key], ref_data[key], color="black", label="_Identity Line")
    plt.xlabel(f"{x_label} ({unit})")
    plt.ylabel(f"{y_label} ({unit})")
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

def create_dataframe(
    ref_data: Dict[str, np.ndarray],
    pred_data: Dict[str, np.ndarray],
    key: str,
    sources: Optional[np.ndarray],
    x_label: str,
    y_label: str,
) -> pd.DataFrame:
    """
    Create a DataFrame from reference and predicted data.
    
    Args:
        ref_data: Dictionary containing reference data
        pred_data: Dictionary containing predicted data
        key: Key to extract the specific data from dictionaries
        sources: Optional data sources
        x_label: Label for x-axis
        y_label: Label for y-axis
    Returns:
        DataFrame containing the reference and predicted data
    """
    df = pd.DataFrame(
        {
            x_label: ref_data[key],
            y_label: pred_data[key],
        }
    )

    if "elements" in ref_data and len(ref_data["elements"]) == len(df):
        df["elements"] = ref_data["elements"]
    elif "elements" in pred_data and len(pred_data["elements"]) == len(df):
        df["elements"] = pred_data["elements"]

    if sources is not None:
        assert len(ref_data[key]) % len(sources) == 0, "Number of sources does not match the number of data points"
        repetitions = len(ref_data[key]) // len(sources)
        df["source"] = np.repeat(sources, repetitions)
    return df

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
    ref_data = extract_data(mace_mols, REF_ENERGY_KEY, REF_FORCES_KEY, REF_CHARGES_KEY)
    MACE_data = extract_data(mace_mols, PRED_ENERGY_KEY, PRED_FORCES_KEY, PRED_CHARGES_KEY)
    
    # Read sources file if provided
    sources = None
    if args.sources is not None:
        with open(args.sources, "r") as f:
            sources: np.ndarray = np.array([line.strip() for line in f.readlines()])
        assert len(sources) == len(mace_mols), f"Number of sources does not match the number of configurations: {len(sources)} != {len(mace_mols)}"
    else:
        sources = None
    
    # Print data statistics
    for name, data in zip(["Ref", "FieldMACE"], [ref_data, MACE_data]):
        for key, value in data.items():
            # Skip non-numeric data
            if isinstance(value, np.ndarray) and value.dtype in (np.float32, np.float64, np.int32, np.int64):
                print(f"{name} {key}: {value.shape} Min Max: {np.min(value): .1f} {np.max(value): .1f}")

    # Use the plot function for each data type
    if PLOT_ENERGY:
        plot_data(
            ref_data,
            MACE_data,
            "energy",
            sources,
            "Ref Energy",
            "FieldMACE Energy",
            ENERGY_UNIT,
            "FieldMACEenergy.png",
        )

    if PLOT_FORCES:
        plot_data(
            ref_data,
            MACE_data,
            "forces",
            sources if sources is not None else ref_data["elements"],
            "Ref Forces",
            "FieldMACE Forces",
            FORCES_UNIT,
            "FieldMACEforces.png",
        )

    if PLOT_CHARGES:
        plot_data(
            ref_data,
            MACE_data,
            "charges",
            sources if sources is not None else ref_data["elements"],
            "Ref Charges",
            "FieldMACE Charges",
            CHARGES_UNIT,
            "MACEcharges.png",
        )


if __name__ == "__main__":
    main()