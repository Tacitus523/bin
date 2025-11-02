#!/usr/bin/env python3
"""
Reads properties from an extended-xyz (.extxyz) file
using ASE, extracts the properties from the ASE Atoms and
plots a scatter and a boxplot grouped by data source.
"""

from typing import Optional
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from ase.io import read

# Plot styling constants
FONTSIZE = 30
LABELSIZE = 18
MARKERSIZE = 18
DPI = 100
FIGSIZE = (20, 10)

# Units
UNIT_ENERGY = "eV"
UNIT_FORCE = r"eV/\AA"
UNIT_ESP = "eV/e"
UNIT_ESP_GRAD = r"eV/e/\AA"

# .info and .array keys
INFO_ENERGY_KEY = "ref_energy"
ARRAY_FORCE_KEY = "ref_forces"
ARRAY_ESP_KEY = "esp"
ARRAY_ESP_GRAD_KEY = "esp_gradient"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot energies (scatter + boxplot) from file or .extxyz")
    ap.add_argument("-f", "--file", type=str, dest="file", required=True, help="File with data to plot (txt or .extxyz)")
    ap.add_argument("-s", "--sources", type=str, dest="data_source_file", required=False, default=None, help="File with data sources (one entry per line)")
    ap.add_argument("--out-prefix", type=str, dest="out_prefix", default="data_points_vs_energy", help="Output filename prefix")
    return ap.parse_args()


def read_energies_from_extxyz(path: Path) -> np.ndarray:
    """Read energies from an extended-xyz file using ASE.

    The function looks for the energy in Atoms.info[INFO_ENERGY_KEY]. If that
    key is missing it will fall back to lowercase 'energy'. Missing values
    are converted to np.nan.
    """
    atoms_list = read(str(path), index=":")
    energies = []
    for atoms in atoms_list:
        info = getattr(atoms, "info", {})
        energy = info.get(INFO_ENERGY_KEY)
        if energy is None:
            # Fallback to common lowercase key
            energy = info.get("energy")
        energies.append(np.nan if energy is None else float(energy))
    return np.array(energies, dtype=float)


def read_properties_from_extxyz(path: Path) -> dict:
    """Read multiple metrics from an extended-xyz file using ASE.

    Returns a dict with keys: 'energy', 'force', 'esp', 'esp_grad'. Each value
    is a 1D numpy array of length n_frames. Missing metrics are returned as None.
    For per-atom arrays (forces, esp arrays), a per-frame scalar is computed:
      - force: RMS magnitude across atoms
      - esp: mean absolute esp across atoms
      - esp_grad: RMS magnitude across atoms
    """
    atoms_list = read(path, index=":")
    energies = []
    forces = []
    esps = []
    esp_grads = []

    for atoms in atoms_list:
        info = getattr(atoms, "info", {})
        arrays = getattr(atoms, "arrays", {})

        # Energy
        if INFO_ENERGY_KEY == "energy":
            energy = atoms.get_potential_energy()
        else:
            energy = info.get(INFO_ENERGY_KEY, None)
        if energy is not None:
            energies.append(energy)

        # Force
        if ARRAY_FORCE_KEY == "forces":
            force = atoms.get_forces()
        else:
            force = arrays.get(ARRAY_FORCE_KEY, None)
        if force is not None:
            forces.append(force)

        esp = arrays.get(ARRAY_ESP_KEY, None)
        if esp is not None:
            esps.append(esp)

        esp_grad = arrays.get(ARRAY_ESP_GRAD_KEY, None)
        if esp_grad is not None:
            esp_grads.append(esp_grad)

    if len(energies) == 0:
        raise ValueError(f"No energies found in {path} using key '{INFO_ENERGY_KEY}'")
    energies = np.array(energies, dtype=float)
    
    if len(forces) == 0:
        raise ValueError(f"No forces found in {path} using key '{ARRAY_FORCE_KEY}'")
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
        "energy": energies,
        "force": forces,
        "force_magnitude": force_magnitudes,
        "esp": esps,
        "esp_grad": esp_grads,
        "esp_grad_magnitude": esp_grad_magnitudes,
    }
    return properties


def read_data_sources(path: Path, n_entries: int) -> pd.Series:
    """Read a file with one entry per line. Spaces are kept as part of the entry.

    Returns a pandas Series of length n_entries; if the file has fewer lines it
    will be repeated to match n_entries; if it has more lines it will be truncated.
    """
    # Read lines preserving spaces: use sep='\n'
    df = pd.read_csv(path, header=None, sep="\n", names=["Data Source"])
    series = df["Data Source"]
    if len(series) == n_entries:
        return series.reset_index(drop=True)
    if len(series) < n_entries:
        # Repeat entries to match length
        reps = int(np.ceil(n_entries / len(series)))
        extended = pd.concat([series] * reps, ignore_index=True).iloc[:n_entries]
        return extended.reset_index(drop=True)
    # Truncate if more lines than entries
    return series.iloc[:n_entries].reset_index(drop=True)


def build_dataframes(properties: dict, data_source_file: Optional[str]) -> pd.DataFrame:
    """Build a DataFrames from the metrics dict. Each property according to their shapes.

    metrics keys:
        'energy': np.ndarray (n,)
        'force': np.ndarray (n * natoms * 3)
        'force_magnitude': np.ndarray (n * natoms)
        'esp': list of np.ndarray (n * natoms)
        'esp_grad': np.ndarray (n * natoms * 3)
        'esp_grad_magnitude': np.ndarray (n * natoms)

    Returns a list of DataFrames for each property group.
    """
    dfs = []
    for keys in [("energy",), ("force_magnitude", "esp", "esp_grad_magnitude"), ("force", "esp_grad")]:
        property_dict = {key: properties[key] for key in keys if properties[key] is not None}
        df_part = pd.DataFrame(property_dict)

        if data_source_file is not None:
            n_rows = df_part.shape[0]
            ds = read_data_sources(Path(data_source_file), n_rows)
            df_part["Data Source"] = ds
        else:
            df_part["Data Source"] = ["Unknown Source"] * df_part.shape[0]
        df_part["Data Source"] = df_part["Data Source"].astype("category")
        df_part["Index"] = df_part.index
        dfs.append(df_part)
            
    return dfs

def plot_scatter(data: pd.DataFrame, column: str, ylabel: str, out_file: str) -> None:
    if data.empty or column not in data.columns:
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.scatterplot(data=data, x="Index", y=column, hue="Data Source", marker=".")
    ax.set_xlabel("Data point")
    ax.set_ylabel(ylabel)
    plt.tick_params(axis="both", which="major")
    legend = ax.legend(bbox_to_anchor=(1.05, 1))
    for legend_handle in legend.legend_handles:
        legend_handle.set_alpha(1)
        legend_handle.set_markersize(MARKERSIZE)

    plt.tight_layout()
    plt.savefig(out_file, dpi=DPI)
    plt.close()


def plot_boxplot(data: pd.DataFrame, column: str, ylabel: str, out_file: str) -> None:
    if data.empty or column not in data.columns:
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.boxplot(data=data, x="Data Source", y=column)
    ax.set_xlabel("Data Source")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=45)
    plt.tick_params(axis="both", which="major")
    plt.tight_layout()
    plt.savefig(out_file, dpi=DPI)
    plt.close()


def main() -> None:
    args = parse_args()
    path = Path(args.file)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    properties = read_properties_from_extxyz(path)

    mol_df, atom_df, vector_df = build_dataframes(properties, args.data_source_file)

    sns.set_context("talk")
    # Energy plots
    plot_scatter(mol_df, "energy", f"Energy ({UNIT_ENERGY})", f"{args.out_prefix}_energy.png")
    plot_boxplot(mol_df, "energy", f"Energy ({UNIT_ENERGY})", f"{args.out_prefix}_energy_boxplot.png")

    # Force plots
    plot_scatter(vector_df, "Force", f"Force ({UNIT_FORCE})", f"{args.out_prefix}_force.png")
    plot_boxplot(vector_df, "Force", f"Force ({UNIT_FORCE})", f"{args.out_prefix}_force_boxplot.png")

    # Force magnitude plots
    plot_scatter(atom_df, "force_magnitude", f"Force Magnitude ({UNIT_FORCE})", f"{args.out_prefix}_force_magnitude.png")
    plot_boxplot(atom_df, "force_magnitude", f"Force Magnitude ({UNIT_FORCE})", f"{args.out_prefix}_force_magnitude_boxplot.png")

    # ESP plots
    plot_scatter(atom_df, "ESP", f"ESP ({UNIT_ESP})", f"{args.out_prefix}_esp.png")
    plot_boxplot(atom_df, "ESP", f"ESP ({UNIT_ESP})", f"{args.out_prefix}_esp_boxplot.png")


    # ESP gradient plots
    plot_scatter(vector_df, "ESP_grad", f"ESP gradient ({UNIT_ESP_GRAD})", f"{args.out_prefix}_esp_grad.png")
    plot_boxplot(vector_df, "ESP_grad", f"ESP gradient ({UNIT_ESP_GRAD})", f"{args.out_prefix}_esp_grad_boxplot.png")
    
    # ESP gradient magnitude plots
    plot_scatter(atom_df, "esp_grad_magnitude", f"ESP Gradient Magnitude ({UNIT_ESP_GRAD})", f"{args.out_prefix}_esp_grad_magnitude.png")
    plot_boxplot(atom_df, "esp_grad_magnitude", f"ESP Gradient Magnitude ({UNIT_ESP_GRAD})", f"{args.out_prefix}_esp_grad_magnitude_boxplot.png")

    print(f"Plots saved with prefix '{args.out_prefix}_*'")

if __name__ == "__main__":
    main()