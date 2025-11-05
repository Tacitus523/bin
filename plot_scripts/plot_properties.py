#!/usr/bin/env python3
"""
Reads properties from an extended-xyz (.extxyz) file
using ASE, extracts the properties from the ASE Atoms and
plots a scatter and a boxplot grouped by data source.
"""

from typing import Optional
import argparse
from pathlib import Path
from typing import Optional, List, Dict

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns
import warnings
import numpy as np
import pandas as pd
from ase.io import read

# Plot styling constants
MARKERSIZE = 25
DPI = 100
FIGSIZE = (16, 9)

# Units
UNIT_ENERGY = "eV"
UNIT_FORCE = r"eV/$\AA$"
UNIT_ESP = "eV/e"
UNIT_ESP_GRAD = r"eV/e/$\AA$"

# .info and .array keys
INFO_ENERGY_KEY = "ref_energy"
INFO_DATA_SOURCE_KEY = "data_source"
ARRAY_FORCE_KEY = "ref_force"
ARRAY_ESP_KEY = "esp"
ARRAY_ESP_GRAD_KEY = "esp_gradient"

SUFFIX1 = "vacuum"
SUFFIX2 = "env"
SUFFIX3 = "diff"

PALETTE = sns.color_palette("tab10")
PALETTE.pop(3)  # Remove red color

# Silence seaborn UserWarning about palette length
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"The palette list has more values .* than needed .*",
)

assert SUFFIX1 != SUFFIX2 != SUFFIX3, "Suffixes for output files must be unique."

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot energies (scatter + boxplot) from file or .extxyz")
    ap.add_argument("-f", "--vac_file", type=str, dest="file", required=True, help="File with data to plot (.extxyz)")
    ap.add_argument("-f2", "--env_file", type=str, dest="file2", required=False, default=None, help="Second file for comparison (.extxyz)")
    ap.add_argument("-s", "--sources", type=str, dest="data_source_file", required=False, default=None, help="File with data sources (one entry per line)")
    ap.add_argument("--out-prefix", type=str, dest="out_prefix", default="property", help="Output filename prefix")
    return ap.parse_args()

def read_properties_from_extxyz(path: Path) -> Dict[str, np.ndarray]:
    """Read multiple metrics from an extended-xyz file using ASE.

    Returns a dict with keys: 'energy', 'force', 'esp', 'esp_grad'. Each value
    is a 1D numpy array of length n_frames. Missing metrics are returned as None.
    For per-atom arrays (forces, esp arrays), a per-frame scalar is computed:
      - force: RMS magnitude across atoms
      - esp: mean absolute esp across atoms
      - esp_grad: RMS magnitude across atoms
    """
    atoms_list = read(path, index=":")
    mol_idxs = []
    atom_nums = []
    energies = []
    data_sources = []
    forces = []
    esps = []
    esp_grads = []

    for mol_idx, atoms in enumerate(atoms_list):
        info = getattr(atoms, "info", {})
        arrays = getattr(atoms, "arrays", {})

        mol_idxs.append(mol_idx)
        atom_nums.append(len(atoms))

        # Energy
        if INFO_ENERGY_KEY == "energy":
            energy = atoms.get_potential_energy()
        else:
            energy = info.get(INFO_ENERGY_KEY, None)
        if energy is not None:
            energies.append(energy)
        else:
            raise ValueError(f"Energy key '{INFO_ENERGY_KEY}' not found in Atoms.info for frame {mol_idx}")

        # Data source
        data_source = info.get(INFO_DATA_SOURCE_KEY, "Unknown Source")
        data_sources.append(data_source)

        # Force
        if ARRAY_FORCE_KEY == "forces":
            force = atoms.get_forces()
        else:
            force = arrays.get(ARRAY_FORCE_KEY, None)
        if force is not None:
            forces.append(force)
        else:
            raise ValueError(f"Force key '{ARRAY_FORCE_KEY}' not found in Atoms.arrays for frame {mol_idx}")

        esp = arrays.get(ARRAY_ESP_KEY, None)
        if esp is not None:
            esps.append(esp)

        esp_grad = arrays.get(ARRAY_ESP_GRAD_KEY, None)
        if esp_grad is not None:
            esp_grads.append(esp_grad)

    mol_idxs = np.array(mol_idxs, dtype=int)
    atom_nums = np.array(atom_nums, dtype=int)
    energies = np.array(energies, dtype=float)

    force_magnitudes = np.concatenate([np.linalg.norm(f, axis=1) for f in forces], axis=0)
    forces = np.concatenate([forces.flatten() for forces in forces], axis=0)
    
    if len(esps) == 0:
        esps = None
    else:
        esps = np.concatenate([esp for esp in esps], axis=0)

    if len(esp_grads) == 0:
        esp_grads = None
        esp_grad_magnitudes = None
    else:
        esp_grad_magnitudes = np.concatenate([np.linalg.norm(eg, axis=1) for eg in esp_grads], axis=0)
        esp_grads = np.concatenate([esp_grad.flatten() for esp_grad in esp_grads], axis=0)

    properties = {
        "mol_index": mol_idxs,
        "atom_num": atom_nums,
        "energy": energies,
        "data_source": data_sources,
        "force": forces,
        "force_magnitude": force_magnitudes,
        "esp": esps,
        "esp_grad": esp_grads,
        "esp_grad_magnitude": esp_grad_magnitudes,
    }
    return properties

def read_data_sources(path: Path) -> np.ndarray:
    """Read a file with one entry per line. Spaces are kept as part of the entry.

    Returns a pandas Series with the data sources.
    """
    # Read lines preserving spaces, non-sense delimiter
    df = pd.read_csv(path, header=None, sep="%", names=["Data Source"])
    return  df["Data Source"].to_numpy()

def build_dataframes(properties: dict, data_source_file: Optional[str]) -> List[pd.DataFrame]:
    """Build a DataFrames from the metrics dict. Each property according to their shapes.

    metrics keys:
        'energy': np.ndarray (n,)
        'force': np.ndarray (n * natoms * 3)
        'force_magnitude': np.ndarray (n * natoms)
        'esp': list of np.ndarray (n * natoms)
        'esp_grad': np.ndarray (n * natoms * 3)
        'esp_grad_magnitude': np.ndarray (n * natoms)

    utility keys:
        'mol_index': np.ndarray (n,)
        'atom_num': np.ndarray (n,)

    Returns a list of DataFrames for each property group.
    """
    mol_idxs = properties["mol_index"]
    n_atoms = properties["atom_num"]
    data_sources = properties["data_source"]
    if data_source_file is not None:
        data_sources = read_data_sources(Path(data_source_file))
    
    dfs = []
    for i, keys in enumerate([
        ("energy",), 
        ("force_magnitude", "esp", "esp_grad_magnitude"), 
        ("force", "esp_grad")
        ]):

        property_dict = {key: properties[key] for key in keys if properties[key] is not None}
        df_part = pd.DataFrame(property_dict)
        print(f"Building DataFrame part with columns: {list(df_part.columns)}")
        
        match i:
            case 0:
                df_part["Data Source"] = data_sources
                df_part["Molecule Index"] = mol_idxs
            case 1:
                
                df_part["Data Source"] = np.repeat(data_sources, n_atoms)
                df_part["Molecule Index"] = np.repeat(mol_idxs, n_atoms)
                df_part["Atom Index"] = np.concatenate([np.arange(n) for n in n_atoms], axis=0)
            case 2:
                df_part["Data Source"] = np.repeat(data_sources, n_atoms * 3)
                df_part["Molecule Index"] = np.repeat(mol_idxs, n_atoms * 3)
                df_part["Atom Index"] = np.concatenate([np.repeat(np.arange(n), 3) for n in n_atoms], axis=0)

        categories = pd.api.types.CategoricalDtype(categories=df_part["Data Source"].unique(), ordered=True)
        df_part["Data Source"] = df_part["Data Source"].astype(categories)
        df_part["Index"] = df_part.index

        dfs.append(df_part)
            
    return dfs

def plot_strip(data: pd.DataFrame, column: str, ylabel: str, out_file: str) -> None:
    return
    if data.empty or column not in data.columns:
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.stripplot(data=data, y=column, x="Data Source", hue="Data Source", marker=".",
        palette=PALETTE, order=data["Data Source"].cat.categories, size=4, jitter=True, dodge=False, alpha=0.5)
    ax.set_xlabel("Data Source")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=30)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(-4, 4))
    # legend = ax.legend(bbox_to_anchor=(1.05, 1))
    # for legend_handle in legend.legend_handles:
    #     legend_handle.set_alpha(1)
    #     legend_handle.set_markersize(MARKERSIZE)

    plt.tight_layout()
    plt.savefig(out_file, dpi=DPI)
    plt.close()

def plot_boxplot(data: pd.DataFrame, column: str, ylabel: str, out_file: str) -> None:
    if data.empty or column not in data.columns:
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.boxplot(data=data, x="Data Source", y=column, showfliers=False,
        hue="Data Source", palette=PALETTE, order=data["Data Source"].cat.categories)
    ax.set_xlabel("Data Source")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=30)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(-4, 4), useOffset=False)

    plt.tick_params(axis="both", which="major")
    plt.tight_layout()
    plt.savefig(out_file, dpi=DPI)
    plt.close()

def compute_differences(properties1: dict, properties2: dict) -> dict:
    """Compute differences between two property dictionaries.
    
    Returns a new properties dict with differences (properties2 - properties1).
    """
    diff_properties = {}
    
    # Copy metadata from first file
    diff_properties["mol_index"] = properties1["mol_index"].copy()
    diff_properties["atom_num"] = properties1["atom_num"].copy() 
    diff_properties["data_source"] = properties1["data_source"].copy()
    
    # Get keys from intersection of both property dicts minus metadata keys
    common_keys = set(properties1.keys()).intersection(set(properties2.keys())) - {"mol_index", "atom_num", "data_source"}

    # Compute differences for each property
    for key in common_keys:
        if properties1.get(key) is not None and properties2.get(key) is not None:
            diff_properties[key] = properties2[key] - properties1[key]
        else:
            diff_properties[key] = None
            
    return diff_properties

def main() -> None:
    args = parse_args()
    path1 = Path(args.file)
    if not path1.exists():
        raise FileNotFoundError(f"Input file not found: {path1}")

    properties1 = read_properties_from_extxyz(path1)
    mol_df1, atom_df1, vector_df1 = build_dataframes(properties1, args.data_source_file)
    datasets = [(mol_df1, atom_df1, vector_df1, SUFFIX1)]
    
    # Check if second file is provided
    if args.file2 is not None:
        path2 = Path(args.file2)
        if not path2.exists():
            raise FileNotFoundError(f"Second input file not found: {path2}")
        properties2 = read_properties_from_extxyz(path2)
        
        # Compute differences
        diff_properties = compute_differences(properties1, properties2)
        
        # Build dataframes for all three datasets
        mol_df2, atom_df2, vector_df2 = build_dataframes(properties2, args.data_source_file)
        mol_df_diff, atom_df_diff, vector_df_diff = build_dataframes(diff_properties, args.data_source_file)
        
        datasets.append((mol_df2, atom_df2, vector_df2, SUFFIX2))
        datasets.append((mol_df_diff, atom_df_diff, vector_df_diff, SUFFIX3))

    sns.set_context("talk")
    
    # Generate plots for each dataset
    for mol_df, atom_df, vector_df, suffix in datasets:
        prefix = f"{args.out_prefix}_{suffix}" if args.file2 is not None else args.out_prefix
        
        # Energy plots
        plot_strip(mol_df, "energy", f"Energy ({UNIT_ENERGY})", f"{prefix}_energy_stripplot.png")
        plot_boxplot(mol_df, "energy", f"Energy ({UNIT_ENERGY})", f"{prefix}_energy_boxplot.png")

        # Force plots
        plot_strip(vector_df, "force", f"Force ({UNIT_FORCE})", f"{prefix}_force_stripplot.png")
        plot_boxplot(vector_df, "force", f"Force ({UNIT_FORCE})", f"{prefix}_force_boxplot.png")

        # Force magnitude plots
        plot_strip(atom_df, "force_magnitude", f"Force Magnitude ({UNIT_FORCE})", f"{prefix}_force_magnitude_stripplot.png")
        plot_boxplot(atom_df, "force_magnitude", f"Force Magnitude ({UNIT_FORCE})", f"{prefix}_force_magnitude_boxplot.png")

        # ESP plots
        plot_strip(atom_df, "esp", f"ESP ({UNIT_ESP})", f"{prefix}_esp_stripplot.png")
        plot_boxplot(atom_df, "esp", f"ESP ({UNIT_ESP})", f"{prefix}_esp_boxplot.png")

        # ESP gradient plots
        plot_strip(vector_df, "esp_grad", f"ESP gradient ({UNIT_ESP_GRAD})", f"{prefix}_esp_grad_stripplot.png")
        plot_boxplot(vector_df, "esp_grad", f"ESP gradient ({UNIT_ESP_GRAD})", f"{prefix}_esp_grad_boxplot.png")
        
        # ESP gradient magnitude plots
        plot_strip(atom_df, "esp_grad_magnitude", f"ESP Gradient Magnitude ({UNIT_ESP_GRAD})", f"{prefix}_esp_grad_magnitude_stripplot.png")
        plot_boxplot(atom_df, "esp_grad_magnitude", f"ESP Gradient Magnitude ({UNIT_ESP_GRAD})", f"{prefix}_esp_grad_magnitude_boxplot.png")

    if args.file2 is not None:
        print(f"Plots saved with prefixes '{args.out_prefix}_{SUFFIX1}_*', '{args.out_prefix}_{SUFFIX2}_*', and '{args.out_prefix}_{SUFFIX3}_*'")
    else:
        print(f"Plots saved with prefix '{args.out_prefix}_*'")

if __name__ == "__main__":
    main()