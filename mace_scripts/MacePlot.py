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

MACE_GEOMS: str = "geoms_mace.extxyz"  # Should already contain reference and MACE data
DATA_SOURCES_FILE: Optional[str] = None  # File containing the data source of each entry

PLOT_CHARGES: bool = True
PLOT_ENERGY: bool = True
PLOT_FORCES: bool = True
PLOT_DMA: bool = False

# Keywords for extracting data
REF_ENERGY_KEY: str = "ref_energy"
REF_FORCES_KEY: str = "ref_force"
REF_CHARGES_KEY: str = "ref_charge"
REF_DMA_KEY: str = "ref_multipoles"
PRED_ENERGY_KEY: str = "MACE_energy"
PRED_FORCES_KEY: str = "MACE_forces"
PRED_CHARGES_KEY: str = "MACE_charges"
PRED_DMA_KEY: str = "MACE_multipoles"

# Units for plotting
ENERGY_UNIT: str = "eV"
FORCES_UNIT: str = r"$\frac{eV}{\AA}$"
CHARGES_UNIT: str = "e"
DMA_UNIT: str = r"$\frac{e}{\AA^3}$"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plotting script for MACE data")
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
    parser.add_argument(
        "-e",
        "--energy",
        action="store_false",
        default=PLOT_ENERGY,
        help="Plot energy data, default: %s" % PLOT_ENERGY,
    )
    parser.add_argument(
        "-f",
        "--forces",
        action="store_false",
        default=PLOT_FORCES,
        help="Plot forces data, default: %s" % PLOT_FORCES,
    )
    parser.add_argument(
        "-c",
        "--charges",
        action="store_false",
        default=PLOT_CHARGES,
        help="Plot charges data, default: %s" % PLOT_CHARGES,
    )
    parser.add_argument(
        "-d",
        "--DMA",
        action="store_false",
        default=PLOT_DMA,
        help="Plot DMA data, default: %s" % PLOT_DMA,
    )
    args = parser.parse_args()
    return args

def extract_data(
    mols: List[Atoms],
    energy_keyword: Optional[str] = None,
    forces_keyword: Optional[str] = None,
    charges_keyword: Optional[str] = None,
    DMA_keyword: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    ref_energy: List[float] = []
    ref_forces: List[float] = []
    ref_charges: List[float] = []
    ref_DMA: List[float] = []
    ref_elements: List[str] = []
    for m in mols:
        if charges_keyword is not None:
            if charges_keyword == "charge":
                ref_charges.extend(m.get_charges())
            else:
                ref_charges.extend(m.arrays[charges_keyword])
        if energy_keyword is not None:
            if energy_keyword == "energy":
                ref_energy.append(m.get_potential_energy())
            else:
                ref_energy.append(m.info[energy_keyword])
        if forces_keyword is not None:
            if forces_keyword == "forces":
                ref_forces.extend(m.get_forces().flatten())
            else:
                ref_forces.extend(m.arrays[forces_keyword].flatten())
        if DMA_keyword is not None:
            AIMS_atom_multipoles = m.arrays[DMA_keyword]
            ref_DMA.extend(AIMS_atom_multipoles[:, 0])
        ref_elements.extend(m.get_chemical_symbols())
    result = {}
    result["energy"] = np.array(ref_energy)  # Energy in eV
    result["forces"] = np.array(ref_forces)  # Forces in eV/Å
    if len(ref_charges) > 0:
        result["charges"] = np.array(ref_charges)
    if len(ref_DMA) > 0:
        result["DMA"] = np.array(ref_DMA)
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
        sources: Data sources
        x_label: Label for x-axis
        y_label: Label for y-axis
        filename: Output filename for the plot
    """
    df: pd.DataFrame = create_dataframe(ref_data, pred_data, key, sources, x_label, y_label)
    rmse: float = np.sqrt(np.mean((df[x_label] - df[y_label]) ** 2))
    r2: float = df[x_label].corr(df[y_label], method="pearson") ** 2

    sns.set_context("talk")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x=x_label,
        y=y_label,
        hue="source" if sources is not None else None,
        palette="viridis",
        alpha=0.7,
        edgecolor=None,
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
    plt.legend(title=None, loc="upper left", fontsize="small")
    for legend_handle in ax.get_legend().legend_handles:
        legend_handle.set_alpha(1)
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
    if sources is not None:
        assert len(ref_data[key]) % len(sources) == 0, "Number of sources does not match the number of data points"
        repetitions = len(ref_data[key]) // len(sources)
        df["source"] = np.repeat(sources, repetitions)
    return df

def main() -> None:
    args: argparse.Namespace = parse_args()
    if args.energy:
        ref_energy_key: Optional[str] = REF_ENERGY_KEY
        pred_energy_key: Optional[str] = PRED_ENERGY_KEY
    else:
        ref_energy_key = None
        pred_energy_key = None
    if args.forces:
        ref_forces_key: Optional[str] = REF_FORCES_KEY
        pred_forces_key: Optional[str] = PRED_FORCES_KEY
    else:
        ref_forces_key = None
        pred_forces_key = None
    if args.charges:
        ref_charges_key: Optional[str] = REF_CHARGES_KEY
        pred_charges_key: Optional[str] = PRED_CHARGES_KEY
    else:
        ref_charges_key = None
        pred_charges_key = None
    if args.DMA:
        ref_dma_key: Optional[str] = REF_DMA_KEY
        pred_dma_key: Optional[str] = PRED_DMA_KEY
    else:
        ref_dma_key = None
        pred_dma_key = None
    mace_mols: List[Atoms] = read(args.geoms, format="extxyz", index=":")
    ref_data: Dict[str, np.ndarray] = extract_data(
        mace_mols, ref_energy_key, ref_forces_key, ref_charges_key, ref_dma_key
    )
    MACE_data: Dict[str, np.ndarray] = extract_data(
        mace_mols, pred_energy_key, pred_forces_key, pred_charges_key, pred_dma_key
    )
    assert len(ref_data["energy"]) == len(mace_mols), "Number of reference data does not match the number of configurations"
    assert len(MACE_data["energy"]) == len(mace_mols), "Number of MACE data does not match the number of configurations"

    if args.sources is not None:
        with open(args.sources, "r") as f:
            sources: np.ndarray = np.array([line.strip() for line in f.readlines()])
        assert len(sources) == len(mace_mols), f"Number of sources does not match the number of configurations: {len(sources)} != {len(mace_mols)}"
    else:
        sources = None

    for name, data in zip(["Ref", "MACE"], [ref_data, MACE_data]):
        for key, value in data.items():
            # Skip non-numeric data
            if value.dtype not in (np.float32, np.float64, np.int32, np.int64):
                continue
            print(f"{name} {key}: {value.shape} Min Max: {np.min(value): .1f} {np.max(value): .1f}")

    # Use the plot function for each data type
    if args.energy:
        plot_data(
            ref_data,
            MACE_data,
            "energy",
            sources,
            "Ref Energy",
            "Mace Energy",
            ENERGY_UNIT,
            "MACEenergy.png",
        )

    if args.forces:
        plot_data(
            ref_data,
            MACE_data,
            "forces",
            sources if sources is not None else ref_data["elements"],
            "Ref Forces",
            "Mace Forces",
            FORCES_UNIT,
            "MACEforces.png",
        )

    if args.DMA:
        plot_data(
            ref_data,
            MACE_data,
            "DMA",
            sources,
            "Ref DMA",
            "Mace DMA",
            DMA_UNIT,
            "MACEdma.png",
        )

    if args.charges:
        plot_data(
            ref_data,
            MACE_data,
            "charges",
            sources if sources is not None else ref_data["elements"],
            "Hirshfeld Charges",
            "Mace Charges",
            CHARGES_UNIT,
            "MACEcharges.png",
        )

if __name__ == "__main__":
    main()