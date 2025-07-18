#!/usr/bin/env python
# Intended use: comparison of forces from .xvgs from reruns and .extxyz files 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pandas as pd
from typing import Dict, List, Optional, Tuple
from ase import Atoms
from ase.io import read

ENERGY_KEY = "ref_energy"
FORCES_KEY = "ref_force"

ENERGY_UNIT = "eV"  # Unit for energies
FORCE_UNIT = "eV/Å"  # Unit for forces

MAX_DATA_POINTS = 25000  # Maximum number of data points to plot

dftb_atomic_energies = {
    1: -7.609986074389834,
    6: -39.29249996225988,
    7: -60.326270220805434,
    8: -85.49729667072424,
    16:-79.51527014111296
} # eV

dft_atomic_energies = {
    1: -13.575035506869515,
    6: -1029.6173622986487,
    7: -1485.1410643783852,
    8: -2042.617308911902,
    16: -10832.265333248919
} # eV


H_connectivity = {
    1: 0,
    2: 0,
    3: 0,
    7: 6,
    11: 10,
    12: 10,
    13: 10,
    17: 16,
    19: 18,
    20: 18,
    21: 20,
}

nm_TO_angstrom = 10.0  # Conversion factor from nm to Ångstrom
eV_TO_kJ_per_mol = 96.485  # Conversion factor from eV to kJ/mol
kJ_mol_nm_TO_eV_angstrom = (1/eV_TO_kJ_per_mol) / nm_TO_angstrom  # Conversion factor from kJ/mol/nm to eV/Å

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compare and plot flattened XVG data')
    parser.add_argument('dftfile', type=str, help='Path to DFT .extxyz file')
    parser.add_argument('dftbfile', type=str, help='Path to DFTB .extxyz file')
    parser.add_argument('-s', '--source', type=str, help='Source of the data', default=None)
    parser.add_argument('--output_prefix', '-o', type=str, help='Output file path', default='correlation')
    parser.add_argument('--labels', '-l', type=str, nargs=2, help='Labels for datasets', 
                        default=['DFT', 'DFTB'])
    parser.add_argument('--fig-size', type=float, nargs=2, help='Figure size (width, height)', 
                        default=[10, 8])
    parser.add_argument('--alpha', type=float, help='Point transparency', default=0.3)
    parser.add_argument('--point-size', type=float, help='Point size', default=10)
    args = parser.parse_args()
    return args

def read_extxyz_file(file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        molecules: List[Atoms] = read(file_path, format='extxyz', index=':')
    except Exception as e:
        raise ValueError(f"Error reading EXTXYZ file {file_path}: {e}")
    
    if not molecules:
        raise ValueError(f"No valid data found in {file_path}")
    
    # Extract forces from the Atoms objects
    atomic_numbers = []
    elements = []
    energies = []
    forces = []
    for molecule in molecules:
            atomic_numbers.append(molecule.get_atomic_numbers())
            elements.append(molecule.get_chemical_symbols())
            energies.append(molecule.info[ENERGY_KEY])
            forces.append(molecule.arrays[FORCES_KEY].flatten())
    atomic_numbers = np.array(atomic_numbers)
    elements = np.array(elements)
    energies = np.array(energies)
    forces = np.array(forces)

    return atomic_numbers, elements, energies, forces

def calculate_statistics(data1: np.ndarray, data2: np.ndarray) -> Dict[str, float]:
    """
    Calculate statistical measures between two datasets.
    
    Args:
        data1: First dataset
        data2: Second dataset
        
    Returns:
        Dictionary containing correlation coefficient, RMSE, and R²
    """
    corr = np.corrcoef(data1, data2)[0, 1]
    rmse = np.sqrt(np.mean((data1 - data2)**2))
    # R² calculation
    ss_tot = np.sum((data2 - np.mean(data2))**2)
    ss_res = np.sum((data2 - data1)**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return {
        'correlation': corr,
        'rmse': rmse,
        'r_squared': r_squared
    }

def calculate_bond_energies(
        energies: np.ndarray, 
        atomic_numbers: np.ndarray, 
        atomic_energies: Dict[int, float]
    ) -> np.ndarray:
    """
    Calculate bond energies by subtracting atomic energies from total energies.
    
    Args:
        energies: Total energies for each configuration
        atomic_numbers: Atomic numbers for each configuration 
        atomic_energies: Dictionary mapping atomic numbers to atomic energies
        
    Returns:
        Bond energies (total energy - sum of atomic energies)
    """
    bond_energies = []
    
    for i, energy in enumerate(energies):
        # Sum atomic energies for this configuration
        atomic_energy_sum = sum(atomic_energies[atomic_num] for atomic_num in atomic_numbers[i])
        
        # Calculate bond energy
        bond_energy = energy - atomic_energy_sum
        bond_energies.append(bond_energy)
    
    return np.array(bond_energies)

def plot_correlation(
        df: pd.DataFrame, 
        stats: Dict[str, float], 
        labels: List[str],
        output_suffix: str,
        unit: str, 
        args: argparse.Namespace,
        ax: Optional[plt.Axes] = None
    ) -> Tuple[plt.Figure, plt.Axes]:

    # Create new figure/axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=tuple(args.fig_size))
        is_standalone = True
    else:
        fig = ax.get_figure()
        is_standalone = False

    # Subsample the data for plotting
    n_data_points = min(len(df), MAX_DATA_POINTS)
    df = df.sample(n=n_data_points, random_state=42)

    hue = "Element" if "Element" in df.columns else "Source" if "Source" in df.columns else None
    if hue == "Element" and len(df['Element'].unique()) == 1:
        hue = "Atom Index" if "Atom Index" in df.columns else None

    hue_order = None if hue is None else sorted(df[hue].unique())

    plot = sns.scatterplot(
        data=df, 
        x=labels[0], 
        y=labels[1],
        hue=hue,
        hue_order=hue_order,
        palette='tab10' if hue else None,
        alpha=args.alpha,
        s=args.point_size,
        ax=ax
    )

    lim_min = min(df[labels[0]].min(), df[labels[1]].min())
    lim_max = max(df[labels[0]].max(), df[labels[1]].max())
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)         
    
    ax.set_xlabel(f"{labels[0]} ({unit})")
    ax.set_ylabel(f"{labels[1]} ({unit})")
    
    # Add perfect correlation line
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 
             color='red', linestyle='--', linewidth=1, label='Perfect Correlation')
    
    ax.legend(loc='lower right', title=hue)
    legend = plot.get_legend()
    if legend is not None:
        for legend_handle in legend.legend_handles:
            legend_handle.set_markersize(args.point_size)  # Set legend point size
            legend_handle.set_alpha(1.0)  # Set legend point transparency to 1.0

    # Add annotation with statistics
    annotation_text = (
        f'RMSE: {stats["rmse"]:.3f} {unit}\n'
        f'R²: {stats["r_squared"]:.4f}'
    )
    
    ax.annotate(
        annotation_text,
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=1, alpha=0.7),
        va='top'
    )
    
    # Save figure
    if is_standalone is True:
        plt.tight_layout()
        plt.savefig(f"{args.output_prefix}_{output_suffix}.png", dpi=300)
        plt.close(fig)

    return fig, ax

def main() -> None:
    args = parse_args()

    # Read DFT and DFTB data
    print(f"Reading DFT data from {args.dftfile}")
    dft_atomic_numbers, dft_elements, dft_total_energies, dft_forces = read_extxyz_file(args.dftfile)
    print(f"Reading DFTB data from {args.dftbfile}")
    dftb_atomic_numbers, dftb_elements, dftb_total_energies, dftb_forces = read_extxyz_file(args.dftbfile)
    assert np.array_equal(dft_atomic_numbers, dftb_atomic_numbers), "Atomic numbers must match in both datasets."
    assert dft_total_energies.shape == dftb_total_energies.shape, "Both datasets must have the same shape."
    assert dft_forces.shape == dftb_forces.shape, "Both datasets must have the same shape."
    
    if args.source is not None:
        sources = np.loadtxt(args.source, dtype=str, delimiter=',')
        assert len(sources) == dft_forces.shape[0], "Source file must have the same number of entries as the datasets."


    # dft_atomization_energies = calculate_bond_energies(dft_total_energies, dft_atomic_numbers, dft_atomic_energies)
    # dftb_atomization_energies = calculate_bond_energies(dftb_total_energies, dftb_atomic_numbers, dftb_atomic_energies)
    # minimal_dft_energy_idx = np.argmin(dftb_total_energies)
    # relative_dft_total_energies = dft_total_energies - dft_total_energies[minimal_dft_energy_idx]
    # relative_dftb_total_energies = dftb_total_energies - dftb_total_energies[minimal_dft_energy_idx]
    dft_mean = np.mean(dft_total_energies)
    dftb_mean = np.mean(dftb_total_energies)
    # dft_mean = -13489.227541673252
    # dftb_mean = -712.5616871654851
    print(f"DFT Mean Energy: {dft_mean} eV")
    print(f"DFTB Mean Energy: {dftb_mean} eV")
    centered_dft_total_energies = dft_total_energies - dft_mean
    centered_dftb_total_energies = dftb_total_energies - dftb_mean

    dft_forces_flat = dft_forces.flatten()
    dftb_forces_flat = dftb_forces.flatten()

    # Calculate statistics
    energy_stats = calculate_statistics(centered_dft_total_energies, centered_dftb_total_energies)
    force_stats = calculate_statistics(dft_forces_flat, dftb_forces_flat)
    
    # Prepare dataframe for plotting
    # Prepare energy data
    energy_labels = [f"{args.labels[0]} Centered Energy", f"{args.labels[1]} Centered Energy"]
    molecule_df = pd.DataFrame({
        energy_labels[0]: centered_dft_total_energies,
        energy_labels[1]: centered_dftb_total_energies,
    })
    if args.source is not None:
        molecule_df['Source'] = sources
        molecule_df = molecule_df.sort_values(by='Source')
        molecule_df = molecule_df[molecule_df['Source'].isin(["300K Simulation", "500K Simulation", "Halved H-bond Constant Simulation"])]

    force_labels = [f"{args.labels[0]} Force", f"{args.labels[1]} Force"]
    atom_indices = np.tile(np.repeat(np.arange(dft_forces.shape[1]//3), 3), dft_forces.shape[0])

    atom_df = pd.DataFrame({
        force_labels[0]: dft_forces_flat,
        force_labels[1]: dftb_forces_flat,
        "Element": np.repeat(dft_elements.flatten(), 3),
        "Atom Index": atom_indices,
        "Coordinate": np.repeat(["x", "y", "z"], dft_forces.shape[0] * (dft_forces.shape[1] // 3)),
    })
    if args.source is not None:
        atom_df['Source'] = np.repeat(sources, dft_forces.shape[1])  # Repeat sources for each force component
        atom_df = atom_df.sort_values(by='Source')
        atom_df = atom_df[atom_df['Source'].isin(["300K Simulation", "500K Simulation", "Halved H-bond Constant Simulation"])]

    # Confirm if all keys in H_connectivity are Hydrogen atoms
    key_indices_df = atom_df[atom_df["Atom Index"].isin(H_connectivity.keys())]
    if key_indices_df["Element"].eq("H").all():
        # Replace all Hydrogen indices with their bond partners
        atom_df["Atom Index"] = atom_df["Atom Index"].replace(H_connectivity)

    print("Plotting correlations...")
    sns.set_context("talk")
    sns.set_style("whitegrid")
    plot_correlation(molecule_df, energy_stats, energy_labels, "energy", ENERGY_UNIT, args)
    plot_correlation(atom_df, force_stats, force_labels, "force", FORCE_UNIT, args)

    # Create subplot for each element
    n_elements = len(np.unique(dft_elements))
    n_cols = 2
    n_rows = (n_elements + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(args.fig_size[0] * n_cols, args.fig_size[1] * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    for ax, element in zip(axes, np.unique(dft_elements)):
        element_mask = atom_df['Element'] == element
        element_df = atom_df[element_mask]
        element_stats = calculate_statistics(
            element_df[force_labels[0]], 
            element_df[force_labels[1]]
        )
        plot_correlation(
            element_df, 
            element_stats, 
            force_labels, 
            f"force_{element}", 
            FORCE_UNIT, 
            args, 
            ax=ax
        )
        ax.set_title(f"Element: {element}")
    plt.tight_layout()
    plt.savefig(f"{args.output_prefix}_element_forces.png", dpi=300)
    plt.close(fig)

if __name__ == '__main__':
    main()