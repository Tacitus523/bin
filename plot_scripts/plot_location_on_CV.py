#!/usr/bin/env python3

import argparse
import os
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
import matplotlib.colors as mcolors
import seaborn as sns

# GLOBAL CONSTANTS
GEOM_FILE = "geoms.extxyz"
ENERGY_KEY = "ref_energy"
DATA_SOURCE_KEY = "data_source"

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
N_COLORS = 10  # Adjust number of discrete colors
ENERGY_COLORMAP = "coolwarm"
# ENERGY_COLORMAP = mcolors.ListedColormap(plt.cm.get_cmap(ENERGY_COLORMAP)(np.linspace(0, 1, N_COLORS)))
ERROR_COLORMAP = mcolors.ListedColormap(plt.cm.get_cmap(COLORMAP)(np.linspace(0, 1, N_COLORS)))

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

assert ENERGY_KEY != "energy", "ENERGY_KEY should not be set to 'energy' as ase uses 'energy' internally"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot location of 2D CV on CV landscape.")
    parser.add_argument(
        "-g", "--geom", type=str, nargs="+",
        default=[GEOM_FILE], help="Path to .extxyz geometry file, can provide 2 files for comparison. Difference gets plotted if 2 files are provided. Difference = energy_file2 - energy_file1"
    )
    parser.add_argument(
        "-s", "--system", type=str, default="ala", choices=IMPLEMENTED_SYSTEMS, help="System to analyze"
    )
    parser.add_argument(
        "-e", "--energy_key", type=str, default=ENERGY_KEY, help=f"key for energies in Atoms arrays (default: {ENERGY_KEY})"
    )
    parser.add_argument(
        "-i", "--identifier", type=str, nargs="+",
        default=[None], help="Identifier for the dataset (used for label the save files)"
    )
    args = parser.parse_args()

    if len(args.geom) > 2:
        raise ValueError("A maximum of two geometry files can be provided.")
    if not len(args.geom) == len(args.identifier):
        raise ValueError("Number of geometry files and identifiers must be the same.")
    for file in args.geom:
        if not os.path.isfile(file):
            raise FileNotFoundError(f"Geometry file {file} not found.")

    for key, value in vars(args).items():
        print(f"{key}: {value}")
    return args

def prepare_basic_data(args: argparse.Namespace) -> pd.DataFrame:
    molecules: List[ase.Atoms] = read(args.geom, index=":")
    data_sources = []
    energies = []
    for i, atoms in enumerate(molecules):
        source = atoms.info.get(DATA_SOURCE_KEY, "Unknown")
        ref_energy = atoms.info.get(args.energy_key, np.nan)
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
    vmin = plot_kwargs.get("vmin", None)
    vmax = plot_kwargs.get("vmax", None)
    identifier = ""

    scatter_kws = {"hue": hue, "palette": palette}
    # Check if hue is numerical data
    if pd.api.types.is_numeric_dtype(df[hue]):
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
        formatter.set_useOffset(round(vmax, 0) if vmax is not None and abs(vmax) > 1e4 else False)
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

    output_file = f"cv_location_{hue.replace(' ', '_').replace('$', '').replace('\\', '').lower()}{identifier}.png" 
    plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Location plot saved to {output_file}")

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
    palette = plot_kwargs.get("palette", ENERGY_COLORMAP)
    unit = plot_kwargs.get("unit", "")
    identifier = plot_kwargs.get("identifier", "")
    vmin = plot_kwargs.get("vmin", None)
    vmax = plot_kwargs.get("vmax", None)
    reduction_method = plot_kwargs.get("reduction_method", "min")
    implemented_reduction_methods = ["min", "mean", "rms"]
    assert reduction_method in implemented_reduction_methods, f"reduction_method must be either in {implemented_reduction_methods}"

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
                # Note: j,i for correct orientation
                if reduction_method == "min":
                    min_energies[j, i] = bin_energies.min()
                elif reduction_method == "mean":
                    min_energies[j, i] = bin_energies.mean()
                elif reduction_method == "rms":
                    min_energies[j, i] = np.sqrt(np.mean(bin_energies**2))
                else:
                    raise ValueError(f"reduction_method must be either in {implemented_reduction_methods}")
    
    # Create the plot
    X, Y = np.meshgrid(x_bins[:-1], y_bins[:-1])
    
    im = plt.pcolormesh(X, Y, min_energies, cmap=palette, vmin=vmin, vmax=vmax)
    
    plt.xlabel(plot_kwargs["x"])
    plt.ylabel(plot_kwargs["y"])
    plt.xlim(xlim)
    plt.ylim(ylim)
    
    # Add colorbar
    cbar = plt.colorbar(im)
    if reduction_method == "min":
        cbar.set_label(f"Minimal {hue} ({unit})" if unit else f"Minimal {hue}")
    elif reduction_method == "mean":
        cbar.set_label(f"Mean {hue} ({unit})" if unit else f"Mean {hue}")
    elif reduction_method == "rms":
        cbar.set_label(f"RMS {hue} ({unit})" if unit else f"RMS {hue}")
    else:
        raise ValueError(f"reduction_method must be either in {implemented_reduction_methods}")

    # Force scientific notation with offset for large numbers
    formatter = ScalarFormatter(useOffset=True, useMathText=True)
    formatter.set_useOffset(round(vmax, 0) if vmax is not None and abs(vmax) > 1e4 else False)
    cbar.ax.yaxis.set_major_formatter(formatter)

    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()

    output_file = f"cv_2D_energy_bins{identifier}.png"
    plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"2D energy bins plot saved to {output_file}")

def main() -> None:
    args = parse_args()
    files = args.geom
    identifiers = args.identifier
    
    dfs = []
    plot_kwargs_list = []
    for file, identifier in zip(files, identifiers):
        args.geom = file
        args.identifier = identifier
        if args.system == "ala":
            file_df, plot_kwargs = prepare_ala_data(args)
        elif args.system == "tder":
            file_df, plot_kwargs = prepare_tder_data(args)
        else:
            raise ValueError(f"System {args.system} not implemented.")
        dfs.append(file_df)
        plot_kwargs_list.append(plot_kwargs)
    full_df = pd.concat(dfs, ignore_index=True)

    sns.set_context("talk", font_scale=1.2)
    for idx, (df, plot_kwargs) in enumerate(zip(dfs, plot_kwargs_list)):
        energy_plot_kwargs = plot_kwargs.copy()
        energy_plot_kwargs["hue"] = "Energy"
        energy_plot_kwargs["palette"] = ENERGY_COLORMAP
        energy_plot_kwargs["unit"] = ENERGY_UNIT
        # Use colorbar for numerical data with 95th percentile normalization
        energy_plot_kwargs["vmin"] = np.percentile(full_df["Energy"].dropna(), 5)
        energy_plot_kwargs["vmax"] = np.percentile(full_df["Energy"].dropna(), 95)

        # Downsample data if too large
        sub_df = df.sample(n=min(len(df), MAX_DATA_POINTS), random_state=42)

        # Energy independent plots
        if idx == 0:
            plot_2D_histogram(df, plot_kwargs)
            plot_location_on_cv(sub_df, plot_kwargs)

        # Energy dependent plots
        plot_2D_energy_bins(df, energy_plot_kwargs)
        plot_location_on_cv(sub_df, energy_plot_kwargs)

    if len(files) == 1:
        return
    
    # If two files are provided, plot the difference
    df1, df2 = dfs
    identifier1, identifier2 = plot_kwargs_list[0]["identifier"], plot_kwargs_list[1]["identifier"]
    difference_df = df2.copy()
    difference_df[r"$\Delta$Energy"] = df2["Energy"] - df1["Energy"]
    difference_df = difference_df.drop(columns=["Energy"])
    difference_plot_kwargs = energy_plot_kwargs.copy()
    difference_plot_kwargs["identifier"] = f"_Delta{identifier2}{identifier1}"
    difference_plot_kwargs["hue"] = r"$\Delta$Energy"
    vmin = np.percentile(difference_df[r"$\Delta$Energy"].dropna(), 5)
    vmax = np.percentile(difference_df[r"$\Delta$Energy"].dropna(), 95)
    max_value = max(abs(vmin), abs(vmax)) # Symmetric cbar around zero
    difference_plot_kwargs["vmin"] = -max_value
    difference_plot_kwargs["vmax"] = max_value

    sub_df = difference_df.sample(n=min(len(difference_df), MAX_DATA_POINTS), random_state=42)
    plot_location_on_cv(sub_df, difference_plot_kwargs)

    difference_plot_kwargs["vmin"] = 0
    difference_plot_kwargs["vmax"] = max_value
    difference_plot_kwargs["reduction_method"] = "rms"
    difference_plot_kwargs["palette"] = ERROR_COLORMAP

    plot_2D_energy_bins(difference_df, difference_plot_kwargs)


if __name__ == "__main__":
    main()