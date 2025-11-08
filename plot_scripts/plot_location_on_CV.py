#!/usr/bin/env python3

import argparse
from typing import List, Tuple, Optional, Dict
import warnings

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
)

import ase
from ase.io import read
import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Dihedral
from MDAnalysis.analysis.distances import self_distance_array
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns

# GLOBAL CONSTANTS
GEOM_FILE = "geoms.extxyz"

ENERGY_UNIT = "eV"

FIGSIZE = (10, 8)
BINS = 150 # Bins might be outside the plotted data range
ALPHA = 0.4
MARKER_SIZE = 10
DPI = 100

MAX_DATA_POINTS = 35_000

PALETTE = sns.color_palette("tab10")
PALETTE.pop(3) # Remove red color
COLORMAP = "YlOrRd"
ENERGY_COLORMAP = "coolwarm"

# Silence seaborn UserWarning about palette length
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"The palette list has more values .* than needed .*",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"Reader has no dt information, set to .*",
)

IMPLEMENTED_SYSTEMS = ["ala", "tder"]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot location of 2D CV on CV landscape.")
    parser.add_argument(
        "-g", "--geom", type=str, default=GEOM_FILE, help="Path to .extxyz geometry file"
    )
    parser.add_argument(
        "-s", "--system", type=str, default="ala", choices=IMPLEMENTED_SYSTEMS, help="System to analyze"
    )
    parser.add_argument(
        "-i", "--identifier", type=str, default=None, help="Identifier for the dataset (used for label the save files)"
    )
    args = parser.parse_args()
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    return args

def prepare_basic_data(args: argparse.Namespace) -> pd.DataFrame:
    molecules: List[ase.Atoms] = read(args.geom, index=":")
    data_sources = []
    energies = []
    for i, atoms in enumerate(molecules):
        source = atoms.info.get("data_source", "Unknown")
        ref_energy = atoms.info.get("ref_energy", None)
        data_sources.append(source)
        energies.append(ref_energy)

    df = pd.DataFrame({
        "Mol Index": np.arange(len(molecules)),
        "Data Source": data_sources,
        "Energy": energies,
    })
    df["Data Source"] = pd.Categorical(df["Data Source"], categories=df["Data Source"].unique())

    return df

def prepare_ala_data(args: argparse.Namespace) -> pd.DataFrame:

    df = prepare_basic_data(args)

    universe = mda.Universe(args.geom, format="XYZ")
    # Define the atom indices for the two dihedrals (MDAnalysis uses 0-based indexing)
    dihedral_indices_1 = [4, 6, 8, 14]  # atoms 4-6-8-14
    dihedral_indices_2 = [6, 8, 14, 16]  # atoms 6-8-14-16

    # Verify expected elements for the dihedrals
    expected_elements_1 = ["C", "N", "C", "C"]
    expected_elements_2 = ["N", "C", "C", "N"]
    actual_elements_1 = [universe.atoms[i].name for i in dihedral_indices_1]
    actual_elements_2 = [universe.atoms[i].name for i in dihedral_indices_2]
    assert actual_elements_1 == expected_elements_1, f"Dihedral 1 atoms do not match expected elements: {actual_elements_1} vs {expected_elements_1}"
    assert actual_elements_2 == expected_elements_2, f"Dihedral 2 atoms do not match expected elements: {actual_elements_2} vs {expected_elements_2}"

    # Create Dihedral objects
    dih = Dihedral([universe.atoms[dihedral_indices_1], universe.atoms[dihedral_indices_2]])
    dih.run()

    phi = dih.results.angles[:, 0]
    psi = dih.results.angles[:, 1]
    df["CV1"] = phi
    df["CV2"] = psi

    # Create periodic images
    dfs = []
    for CV_label in ["CV1", "CV2"]:
        for offset in [-360, 0, 360]:
            df_offset = df.copy()
            df_offset[CV_label] = df_offset[CV_label] + offset
            dfs.append(df_offset)
    df = pd.concat(dfs, ignore_index=True)

    plot_kwargs = {
        "system": args.system,
        "identifier": "_"+args.identifier.replace(" ", "_").lower() if args.identifier is not None else "",
        "x": r"$\varphi$ ($\degree$)",
        "y": r"$\psi$ ($\degree$)",
        "xlim": (-180, 120),
        "ylim": (-100, 240),
    }

    return df, plot_kwargs
    
def prepare_tder_data(args: argparse.Namespace) -> pd.DataFrame:

    df = prepare_basic_data(args)

    universe = mda.Universe(args.geom, format="XYZ")
    # Define the atom indices for the two distances (MDAnalysis uses 0-based indexing)
    distance_indices_1 = [6, 1]
    distance_indices_2 = [11, 1]
    excpected_elements_1 = ["S", "S"]
    expected_elements_2 = ["S", "S"]
    actual_elements_1 = [universe.atoms[i].name for i in distance_indices_1]
    actual_elements_2 = [universe.atoms[i].name for i in distance_indices_2]
    assert actual_elements_1 == excpected_elements_1, f"Distance 1 atoms do not match expected elements: {actual_elements_1} vs {excpected_elements_1}"
    assert actual_elements_2 == expected_elements_2, f"Distance 2 atoms do not match expected elements: {actual_elements_2} vs {expected_elements_2}"

    all_distances_1 = []
    all_distances_2 = []
    for ts in universe.trajectory:
        dist1 = mda.lib.distances.calc_bonds(
            universe.atoms[[distance_indices_1[0]]],
            universe.atoms[[distance_indices_1[1]]],
        )
        dist2 = mda.lib.distances.calc_bonds(
            universe.atoms[[distance_indices_2[0]]],
            universe.atoms[[distance_indices_2[1]]],
        )
        all_distances_1.append(dist1[0])
        all_distances_2.append(dist2[0])

    all_distances_1 = np.array(all_distances_1)
    all_distances_2 = np.array(all_distances_2)

    df["CV1"] = all_distances_1
    df["CV2"] = all_distances_2

    plot_kwargs = {
        "system": args.system,
        "identifier": "_"+args.identifier.replace(" ", "_").lower() if args.identifier is not None else "",
        "x": "S¹-S² Distance (Å)",
        "y": "S²-S³ Distance (Å)",
        "xlim": (1.8, 8.0),
        "ylim": (1.8, 8.0),
    }

    return df, plot_kwargs

def plot_location_on_cv(df: pd.DataFrame, plot_kwargs: Dict[str, str]) -> None:
    hue = plot_kwargs.get("hue", "Data Source")
    palette = plot_kwargs.get("palette", PALETTE)
    unit = plot_kwargs.get("unit", "")
    identifier = ""

    scatter_kws = {"hue": hue, "palette": palette}
    # Check if hue is numerical data
    if pd.api.types.is_numeric_dtype(df[hue]):
        # Use colorbar for numerical data with 90th percentile normalization
        vmin = df[hue].min()
        vmax = np.percentile(df[hue].dropna(), 90)
        scatter_kws = {'vmin': vmin, 'vmax': vmax, 'c': df[hue], 'cmap': palette}
        identifier = plot_kwargs.get("identifier", "")

    plt.figure(figsize=FIGSIZE)
    ax = sns.scatterplot(
        data=df,
        x="CV1",
        y="CV2",
        s=MARKER_SIZE,
        alpha=ALPHA,
        **scatter_kws
    )
    ax.set_xlabel(plot_kwargs["x"])
    ax.set_ylabel(plot_kwargs["y"])
    ax.set_xlim(plot_kwargs.get("xlim", None))
    ax.set_ylim(plot_kwargs.get("ylim", None))
    
    # Check if hue is numerical data
    if pd.api.types.is_numeric_dtype(df[hue]):
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(f"{hue} ({unit})" if unit else hue)

        # Force scientific notation with offset for large numbers
        formatter = ScalarFormatter(useOffset=True, useMathText=True)
        formatter.set_useOffset(round(vmax, 0) if vmax is not None else 0)
        cbar.ax.yaxis.set_major_formatter(formatter)
        
        # Remove the legend created by seaborn
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    else:
        # Use legend for categorical data
        plt.legend(title=hue, bbox_to_anchor=(1.05, 1), loc='upper left')
        legend = ax.get_legend()
        if legend is not None:
            for legend_handle in legend.legendHandles:
                legend_handle.set_markersize(MARKER_SIZE*1.5)
                legend_handle.set_alpha(1.0)

    output_file = f"cv_location_{hue.replace(' ', '_').lower()}{identifier}.png" 
    plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_file}")

def plot_2D_histogram(df: pd.DataFrame, plot_kwargs: Dict[str, str]) -> None:
    plt.figure(figsize=FIGSIZE)
    ax = sns.histplot(
        data=df,
        x="CV1",
        y="CV2",
        bins=BINS,
        pmax=0.9,
        cmap=COLORMAP,
        cbar=True,
        cbar_kws={'label': 'Count'},
    )
    ax.set_xlabel(plot_kwargs["x"])
    ax.set_ylabel(plot_kwargs["y"])
    ax.set_xlim(plot_kwargs.get("xlim", None))
    ax.set_ylim(plot_kwargs.get("ylim", None))
    plt.grid(True)

    output_file = f"cv_2D_histogram.png"
    plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"2D histogram plot saved to {output_file}")

def plot_2D_energy_bins(df: pd.DataFrame, plot_kwargs: Dict[str, str]) -> None:
    """Plot 2D bins colored by the minimum energy in each bin"""
    hue = plot_kwargs.get("hue", None)
    unit = plot_kwargs.get("unit", "")
    identifier = plot_kwargs.get("identifier", "")

    plt.figure(figsize=FIGSIZE)
    
    # Define bin edges
    xlim = plot_kwargs.get("xlim", (df["CV1"].min(), df["CV1"].max()))
    ylim = plot_kwargs.get("ylim", (df["CV2"].min(), df["CV2"].max()))
    
    nbins = 50  # Number of bins in each dimension
    x_bins = np.linspace(xlim[0], xlim[1], nbins + 1)
    y_bins = np.linspace(ylim[0], ylim[1], nbins + 1)
    
    # Create 2D array to store minimum energies
    min_energies = np.full((nbins, nbins), np.nan)
    
    # Bin the data and find minimum energy in each bin
    for i in range(nbins):
        for j in range(nbins):
            x_mask = (df["CV1"] >= x_bins[i]) & (df["CV1"] < x_bins[i+1])
            y_mask = (df["CV2"] >= y_bins[j]) & (df["CV2"] < y_bins[j+1])
            bin_mask = x_mask & y_mask
            
            if bin_mask.sum() > 0:  # If there are points in this bin
                bin_energies = df.loc[bin_mask, hue]
                min_energies[j, i] = bin_energies.min()  # Note: j,i for correct orientation
    
    # Create the plot
    X, Y = np.meshgrid(x_bins[:-1], y_bins[:-1])
    
    # Use percentile-based normalization for better contrast
    valid_energies = min_energies[~np.isnan(min_energies)]
    if len(valid_energies) > 0:
        vmin = np.min(valid_energies)
        vmax = np.percentile(valid_energies, 90)
    else:
        vmin, vmax = None, None
    
    im = plt.pcolormesh(X, Y, min_energies, cmap=ENERGY_COLORMAP, vmin=vmin, vmax=vmax)
    
    plt.xlabel(plot_kwargs["x"])
    plt.ylabel(plot_kwargs["y"])
    plt.xlim(xlim)
    plt.ylim(ylim)
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label(f"Minimum {hue} ({unit})" if unit else f"Minimum {hue}")

    # Force scientific notation with offset for large numbers
    formatter = ScalarFormatter(useOffset=True, useMathText=True)
    formatter.set_useOffset(round(vmax, 0) if vmax is not None else 0)
    cbar.ax.yaxis.set_major_formatter(formatter)

    plt.grid(True, alpha=0.3)
    
    output_file = f"cv_2D_energy_bins{identifier}.png"
    plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"2D energy bins plot saved to {output_file}")

def main() -> None:
    args = parse_args()

    if args.system == "ala":
        df, plot_kwargs = prepare_ala_data(args)
    elif args.system == "tder":
        df, plot_kwargs = prepare_tder_data(args)
    else:
        raise ValueError(f"System {args.system} not implemented.")
    

    energy_plot_kwargs = plot_kwargs.copy()
    energy_plot_kwargs["hue"] = "Energy"
    energy_plot_kwargs["palette"] = ENERGY_COLORMAP
    energy_plot_kwargs["unit"] = ENERGY_UNIT

    sns.set_context("talk", font_scale=1.2)

    # Sample size independent plots
    plot_2D_histogram(df, plot_kwargs)
    plot_2D_energy_bins(df, energy_plot_kwargs)

    # Downsample data if too large
    df = df.sample(n=min(len(df), MAX_DATA_POINTS), random_state=42)

    # Sample size dependent plots
    plot_location_on_cv(df, plot_kwargs)
    plot_location_on_cv(df, energy_plot_kwargs)

if __name__ == "__main__":
    main()