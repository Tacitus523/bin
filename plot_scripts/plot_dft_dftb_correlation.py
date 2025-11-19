#!/usr/bin/env python
# Intended use: comparison of forces from .xvgs from reruns and .extxyz files
import json
import warnings

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pandas as pd
from typing import Dict, List, Optional, Tuple
from ase import Atoms
from ase.data import atomic_numbers
from ase.io import read
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error

ENERGY_KEY = "ref_energy"
FORCES_KEY = "ref_force"
ESP_KEY: Optional[str] = None # "esp"
ESP_GRAD_KEY: Optional[str] = None # "esp_gradient"

ENERGY_UNIT = "eV"  # Unit for energies
FORCE_UNIT = "eV/Å"  # Unit for forces

MAX_DATA_POINTS = 15000  # Maximum number of data points to plot

FIGSIZE = (12,9)  # Default figure size for plots
DPI = 150  # Default DPI for saved figures

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

IMPLEMENTED_SYSTEMS = [None, 'ala', 'tder']
H_connectivity_dipeptide = {
    1: 0,
    2: 0,
    3: 0,
    7: 6,
    9: 8,
    11: 10,
    12: 10,
    13: 10,
    17: 16,
    19: 18,
    20: 18,
    21: 20,
}
H_connectivity_thiol = {}

element_specifications_dipeptide = {
    4: "C=O",
    14: "C=O",
    7: "H-N",
    9: "H-CA",
    17: "H-N"
}
element_specifications_thiol = {}

nm_TO_angstrom = 10.0  # Conversion factor from nm to Ångstrom
eV_TO_kJ_per_mol = 96.485  # Conversion factor from eV to kJ/mol
kJ_mol_nm_TO_eV_angstrom = (1/eV_TO_kJ_per_mol) / nm_TO_angstrom  # Conversion factor from kJ/mol/nm to eV/Å

PALETTE = sns.color_palette("tab10")
PALETTE.pop(3)  # Remove red color
COLORMAP = "YlOrRd_r"

# Silence seaborn UserWarning about palette length
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"The palette list has more values .* than needed .*",
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compare and plot flattened XVG data')
    parser.add_argument('dftfile', type=str, help='Path to DFT .extxyz file')
    parser.add_argument('dftbfile', type=str, help='Path to DFTB .extxyz file')
    parser.add_argument('-s', '--source', type=str, help='Source of the data', default=None)
    parser.add_argument('--output_prefix', '-o', type=str, help='Output file path', default='correlation')
    parser.add_argument('--labels', '-l', type=str, nargs=2, help='Labels for datasets', 
                        default=['DFT', 'DFTB'])
    parser.add_argument('--fig-size', type=float, nargs=2, help='Figure size (width, height)', 
                        default=FIGSIZE)
    parser.add_argument('--alpha', type=float, help='Point transparency', default=0.5)
    parser.add_argument('--mean_energies', type=str, help='Path to mean energies JSON file', default=None)
    parser.add_argument('--ref_files', type=str, nargs=2, help='Paths to vacuum .extxyz files for energy referencing', default=None)
    parser.add_argument('--index_files', type=str, nargs="*", help='Paths to index files for selecting geometries(folder_index.txt)', default=None)
    parser.add_argument('--system', type=str, help='System name for spefications of elements and connectivity', default=None, choices=IMPLEMENTED_SYSTEMS)
    args = parser.parse_args()
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    if args.system == None:
        args.h_connectivity = {}
        args.element_specifications = {}
    elif args.system == 'ala':
        args.h_connectivity = H_connectivity_dipeptide
        args.element_specifications = element_specifications_dipeptide
    elif args.system == 'tder':
        args.h_connectivity = H_connectivity_thiol
        args.element_specifications = element_specifications_thiol

    if args.index_files is not None:
        if not len(args.index_files) == (2 if args.ref_files is None else 4):
            raise ValueError("Number of index files must match number of data files (2 or 4).")
    
    return args

def read_extxyz_file(file_path: str, indices: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Args:
        file_path: Path to the .extxyz file
        indices: Indices of geometries to read from the file

    Returns:
        Tuple containing:
            - atomic_numbers: Array of atomic numbers for each configuration
            - elements: Array of element symbols for each configuration
            - energies: Array of total energies for each configuration
            - forces: Array of forces for each configuration (flattened)
            - esps: Array of ESPs for each configuration, empty if ESP_KEY is None
            - e_field_magnitudes: Array of electric field magnitudes for each configuration, empty if ESP_GRAD_KEY is None
    """

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
    esps = []
    e_field_magnitudes = []
    next_index = 0 if indices is not None else None
    for i, molecule in enumerate(molecules):
        if next_index is not None:
            if next_index >= len(indices):
                break
            if i < indices[next_index]:
                continue
            elif i == indices[next_index]:
                next_index += 1
            else:
                raise ValueError(f"Index {indices[next_index]} not found in file {file_path}")

        atomic_numbers.append(molecule.get_atomic_numbers())
        elements.append(molecule.get_chemical_symbols())
        energies.append(molecule.info[ENERGY_KEY])
        forces.append(molecule.arrays[FORCES_KEY].flatten())
        if ESP_KEY is not None and ESP_KEY in molecule.arrays:
            esps.append(molecule.arrays[ESP_KEY].flatten())
        if ESP_GRAD_KEY is not None and ESP_GRAD_KEY in molecule.arrays:
            e_field_magnitude = np.linalg.norm(molecule.arrays[ESP_GRAD_KEY], axis=1).flatten() * -1  # Multiply by -1 to get correct direction
            e_field_magnitudes.append(e_field_magnitude)

    atomic_numbers = np.array(atomic_numbers)
    elements = np.array(elements)
    energies = np.array(energies)
    forces = np.array(forces)
    esps = np.array(esps)
    e_field_magnitudes = np.array(e_field_magnitudes)

    return atomic_numbers, elements, energies, forces, esps, e_field_magnitudes

def compare_folder_orders(
        folder_order_files: Optional[List[str]]
    ) -> List[Optional[np.ndarray]]:
    """Compare folder order files and return lists of shared indices."""

    if folder_order_files is None:
        return [None] * 4

    indices_list = []
    for folder_order_file in folder_order_files:
        # Read file - each row is a unique measurement
        indices = pd.read_csv(
            folder_order_file, 
            header=None, 
            names=["method_idx", "folder", "convergence"],
            converters={"convergence": lambda x: x.strip() == "True"} # Used to be " True" instead of "True"
        ).reset_index(names=["total_idx"])
        indices_list.append(indices)

    # Find total_idx values that converged in all files
    all_converged_filter = np.all([indices["convergence"] for indices in indices_list], axis=0)
    all_converged_total_idx = indices_list[0][all_converged_filter]["total_idx"].to_numpy()

    # Get positions in converged-only arrays (these are the indices for property arrays)
    converged_list = []
    not_converged_list = []
    converged_indices_list = []
    for indices in indices_list:
        # Filter to only converged measurements
        converged = indices[indices["convergence"] == True].reset_index(drop=True).reset_index(names=["relative_idx"]).set_index("total_idx")
        # Filter to only not converged measurements
        not_converged = indices[indices["convergence"] == False]

        converged_indices = converged.loc[all_converged_total_idx,:]["relative_idx"].to_numpy()
        converged_list.append(converged)
        not_converged_list.append(not_converged)
        converged_indices_list.append(converged_indices)
    
    # Print statistics
    for i, (indices, converged, not_converged) in enumerate(zip(indices_list, converged_list, not_converged_list), start=1):
        print(f"Dataset {i}:")
        print(f"  Total measurements: {len(indices)}, converged: {len(converged)}, not converged: {len(not_converged)}")
    print
    print(f"Shared converged measurements: {len(all_converged_total_idx)}")
   
    # Save indices delta
    indices_delta = indices_list[0].copy()
    indices_delta["convergence"] = all_converged_filter.astype(bool)
    delta_folder_order_file = "folder_order_delta.txt"
    indices_delta.to_csv(
        delta_folder_order_file, 
        header=False, 
        index=False,
        sep=","
    )

    return converged_indices_list

def relative_root_mean_squared_error(data1: np.ndarray, data2: np.ndarray) -> float:
    target_rms = np.sqrt(np.mean(np.square(data1)))
    delta_rms = np.sqrt(np.mean(np.square(data1 - data2)))
    return delta_rms / (target_rms+1e-8) * 100

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
    rmse = root_mean_squared_error(data1, data2)
    mae = mean_absolute_error(data1, data2)
    r_squared = r2_score(data1, data2)
    rel_rmse = relative_root_mean_squared_error(data1, data2)
    
    return {
        'correlation': corr,
        'rmse': rmse,
        'mae': mae,
        'r_squared': r_squared,
        'rel_rmse': rel_rmse
    }

def calculate_atomization_energies(
        energies: np.ndarray, 
        atomic_numbers: np.ndarray, 
        atomic_energies: Dict[int, float]
    ) -> np.ndarray:
    """
    Calculate atomization energies by subtracting atomic energies from total energies.
    
    Args:
        energies: Total energies for each configuration
        atomic_numbers: Atomic numbers for each configuration 
        atomic_energies: Dictionary mapping atomic numbers to atomic energies
        
    Returns:
        Atomization energies (total energy - sum of atomic energies)
    """
    atomization_energies = []
    
    for i, energy in enumerate(energies):
        # Sum atomic energies for this configuration
        atomic_energy_sum = sum(atomic_energies[atomic_num] for atomic_num in atomic_numbers[i])
        
        # Calculate bond energy
        bond_energy = energy - atomic_energy_sum
        atomization_energies.append(bond_energy)
    
    return np.array(atomization_energies)

def add_perfect_correlation_line(ax: plt.Axes, lim_min: float, lim_max: float) -> None:
    """Add a perfect correlation line (y=x) to the plot."""
    ax.plot(
        [lim_min, lim_max],
        [lim_min, lim_max],
        color='red',
        linestyle='--',
        linewidth=1,
        #label='Perfect Correlation'
    )

def plot_correlation(
        df: pd.DataFrame, 
        stats: Dict[str, float], 
        labels: List[str],
        output_suffix: str,
        unit: str, 
        args: argparse.Namespace,
        hue: Optional[str] = None,
        color: Optional[str] = None,
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

    # Handle hue/color conflict - color takes precedence over hue
    if color is not None:
        hue = None  # Disable hue when color is specified

    # Fallback for hue
    if hue is None and color is None:
        hue = "Element" if "Element" in df.columns else "Data Source" if "Data Source" in df.columns else None
    if hue is not None:
        unique_values = df[hue].unique()
        if len(unique_values) == 1:
            hue = "Data Source" if "Data Source" in df.columns else None
            unique_values = df[hue].unique()
            if len(unique_values) == 1:
                hue = None

    hue_order = None
    if hue is not None:
        if isinstance(unique_values, pd.Categorical):
            hue_order = unique_values.sort_values()
        elif isinstance(unique_values, np.ndarray):
            hue_order = np.sort(unique_values)
        else:
            hue_order = sorted(unique_values)

    scatter_kwargs = {
        "data": df,
        "x": labels[0],
        "y": labels[1],
        "alpha": args.alpha,
        "ax": ax,
        "s": 10
    }

    if hue is not None:
        scatter_kwargs["hue"] = hue
        scatter_kwargs["hue_order"] = hue_order
        scatter_kwargs["palette"] = PALETTE
    if color is not None:
        scatter_kwargs["c"] = df[color]
        scatter_kwargs["cmap"] = COLORMAP

    plot = sns.scatterplot(
        **scatter_kwargs
    )

    if is_standalone is True:
        lim_min = min(df[labels[0]].min(), df[labels[1]].min())
        lim_max = max(df[labels[0]].max(), df[labels[1]].max())
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)         
        add_perfect_correlation_line(ax, lim_min, lim_max)
    
    ax.set_xlabel(f"{labels[0]} ({unit})")
    ax.set_ylabel(f"{labels[1]} ({unit})")
    
    if hue is not None:
        legend = ax.legend(loc='lower right', title=hue, markerscale=3, frameon=True)
        for legend_handle in legend.legend_handles:
            legend_handle.set_alpha(1.0)  # Set legend point transparency to 1.0
            if hasattr(legend_handle, "set_sizes"):
                legend_handle.set_sizes([30])

    ax.grid(True, alpha=0.3)
    # Add annotation with statistics
    annotation_text = (
        f'RMSE: {stats["rmse"]:.2f} {unit}\n'
        f'Rel. RMSE: {stats["rel_rmse"]:.1f} %'
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
        plt.savefig(f"{args.output_prefix}_{output_suffix}.png", dpi=DPI)
        plt.close(fig)

    return fig, ax

def plot_subplots_per_element(
        atom_df: pd.DataFrame,
        force_labels: List[str],
        args: argparse.Namespace,
        hue: Optional[str] = "Element",
    ) -> None:

     # Create subplot for each element
    unique_elements = atom_df['Element'].unique().sort_values()
    n_unique_elements = atom_df['Element'].nunique()
    max_force = max(atom_df[force_labels[0]].max(), atom_df[force_labels[1]].max())
    min_force = min(atom_df[force_labels[0]].min(), atom_df[force_labels[1]].min())
    n_cols = 2
    n_rows = (n_unique_elements + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(args.fig_size[0] * n_cols, args.fig_size[1] * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    for ax, element in zip(axes, unique_elements):
        element_mask = atom_df['Element'] == element
        element_df = atom_df[element_mask].copy()

        # Create specify elements from global variable
        key_indices_df = element_df[element_df["Atom Index"].isin(args.h_connectivity.keys())]
        element_df["Element"] = element_df["Element"].astype(str) # Convert to string to avoid categorical issues
        element_df["Element"] = element_df["Element"].mask(
            element_df["Atom Index"].isin(args.element_specifications.keys()),
            element_df["Atom Index"].map(args.element_specifications)
        )

        # Confirm if all keys in H_connectivity are Hydrogen atoms
        if key_indices_df["Element"].eq("H").all():
            # Replace all Hydrogen indices with their bond partners to reduce hue clutter
            element_df["Atom Index"] = element_df["Atom Index"].replace(args.h_connectivity)

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
            hue=hue,
            ax=ax
        )
        if hue != "Element":
            ax.set_title(f"Element: {element}")
        ax.set_xlim(min_force, max_force)
        ax.set_ylim(min_force, max_force)
        add_perfect_correlation_line(ax, min_force, max_force)

        # Do individual plots for each element
        if hue == "Element":
            plot_correlation(
                element_df, 
                element_stats, 
                force_labels, 
                f"force_{element}_{hue}", 
                FORCE_UNIT, 
                args,
                hue=hue,
                ax=None
            )

    # Hide any unused subplots by making them invisible
    for i in range(len(unique_elements), len(axes)):
        axes[i].set_visible(False)
    plt.tight_layout()
    hue_clean = hue.replace(" ", "_").lower() if hue is not None else ""
    plt.savefig(f"{args.output_prefix}_{hue_clean}_forces.png", dpi=DPI)
    plt.close(fig)

def main() -> None:
    args = parse_args()

    # Read index files if provided
    converged_indices_list: List[Optional[np.ndarray]] = compare_folder_orders(args.index_files)

    # Read DFT and DFTB data
    dft_atomic_numbers, dft_elements, dft_total_energies, dft_forces, _, _ = read_extxyz_file(args.dftfile, converged_indices_list[0])
    dftb_atomic_numbers, dftb_elements, dftb_total_energies, dftb_forces, _, _ = read_extxyz_file(args.dftbfile, converged_indices_list[1])
    assert np.array_equal(dft_atomic_numbers, dftb_atomic_numbers), "Atomic numbers must match in both datasets."
    assert dft_total_energies.shape == dftb_total_energies.shape, "Both datasets must have the same shape."
    assert dft_forces.shape == dftb_forces.shape, "Both datasets must have the same shape."


    if args.ref_files is not None:
        dft_vac_atomic_numbers, _, dft_vac_total_energies, dft_vac_forces, _, _ = read_extxyz_file(args.ref_files[0], converged_indices_list[2])
        dftb_vac_atomic_numbers, _, dftb_vac_total_energies, dftb_vac_forces, _, _  = read_extxyz_file(args.ref_files[1], converged_indices_list[3])
        assert np.array_equal(dft_vac_atomic_numbers, dftb_vac_atomic_numbers), "Vacuum atomic numbers must match in both datasets."
        assert dft_vac_total_energies.shape == dftb_vac_total_energies.shape, "Both vacuum datasets must have the same shape."
        assert dft_vac_total_energies.shape[0] == dft_total_energies.shape[0], "Vacuum datasets must have the same number of entries as the main datasets."
        assert np.array_equal(dft_vac_atomic_numbers, dft_atomic_numbers), "Vacuum atomic numbers must match main dataset atomic numbers."

        # Subtract vacuum energies from total energies
        dft_total_energies -= dft_vac_total_energies
        dftb_total_energies -= dftb_vac_total_energies
        dft_forces -= dft_vac_forces
        dftb_forces -= dftb_vac_forces

    if args.source is not None:
        sources = np.loadtxt(args.source, dtype=str, delimiter=',')
        sources = sources[converged_indices_list[0]] if converged_indices_list[0] is not None else sources
        assert len(sources) == dft_forces.shape[0], "Source file must have the same number of entries as the datasets."

    if args.ref_files is not None and args.mean_energies is not None:
        with open(args.mean_energies, 'r') as f:
            mean_energies = json.load(f)
        # Expected case: mislabeled dft and dftb data
        # actual difference in this case would be dft_vac - dftb_vac vs dft_env - dftb_env
        # with mean difference its (dft_vac-mean_dft_vac) - (dftb_vac-mean_dftb_vac) vs (dft_env - mean_dft_vac) - (dftb_env - mean_dftb_vac)
        # the reference energies were already subtracted above, so we just need to adjust for the mean difference
        # (dft_vac - dftb_vac){happend above} - (mean_dft_vac - mean_dftb_vac){defined here}
        # (dft_env - dftb_env){happend above} - (mean_dft_vac - mean_dftb_vac){defined here}
        
        dft_mean = mean_energies["dft_mean_energy_eV"] - mean_energies["dftb_mean_energy_eV"]
        dftb_mean = dft_mean
    elif args.mean_energies is not None:
        with open(args.mean_energies, 'r') as f:
            mean_energies = json.load(f)
        dft_mean = mean_energies["dft_mean_energy_eV"]
        dftb_mean = mean_energies["dftb_mean_energy_eV"]
    elif args.ref_files is not None:
        dft_mean = 0.0
        dftb_mean = 0.0
    else:
        # dft_atomization_energies = calculate_atomization_energies(dft_total_energies, dft_atomic_numbers, dft_atomic_energies)
        # dftb_atomization_energies = calculate_atomization_energies(dftb_total_energies, dftb_atomic_numbers, dftb_atomic_energies)
        # minimal_dft_energy_idx = np.argmin(dftb_total_energies)
        # relative_dft_total_energies = dft_total_energies - dft_total_energies[minimal_dft_energy_idx]
        # relative_dftb_total_energies = dftb_total_energies - dftb_total_energies[minimal_dft_energy_idx]
        dft_mean = np.mean(dft_total_energies)
        dftb_mean = np.mean(dftb_total_energies)
        with open(f"{args.output_prefix}_mean_energies.json", 'w') as f:
            json.dump({
                "dft_mean_energy_eV": dft_mean,
                "dftb_mean_energy_eV": dftb_mean
            }, f, indent=4)
    
    centered_dft_total_energies = dft_total_energies - dft_mean
    centered_dftb_total_energies = dftb_total_energies - dftb_mean

    dft_forces_flat = dft_forces.flatten()
    dftb_forces_flat = dftb_forces.flatten()

    # Calculate statistics
    energy_stats = calculate_statistics(centered_dft_total_energies, centered_dftb_total_energies)
    force_stats = calculate_statistics(dft_forces_flat, dftb_forces_flat)
    
    # Prepare dataframe for plotting
    # Prepare energy data
    energy_labels = [f"{args.labels[0]} Energy", f"{args.labels[1]} Energy"]
    molecule_df = pd.DataFrame({
        "Molecule Index": np.arange(dft_forces.shape[0]),
        energy_labels[0]: centered_dft_total_energies,
        energy_labels[1]: centered_dftb_total_energies,
    })
    if args.source is not None:
        molecule_df['Data Source'] = sources
        molecule_df['Data Source'] = molecule_df['Data Source'].str.replace(" Constant Simulation", " Constant")
        molecule_df['Data Source'] = pd.Categorical(molecule_df['Data Source'], categories=molecule_df['Data Source'].unique(), ordered=True)
        #molecule_df = molecule_df[molecule_df['Data Source'].isin(["300K Simulation", "500K Simulation", "Halved H-bond Constant Simulation"])]
        # Remove "Simulation" from source names for better legend display

    # if ESP_KEY is not None and dft_esps.size > 0:
    #     esp_filter = np.any(np.abs(dft_esps) > 5, axis=1)
    #     molecule_df["High ESP Magnitude"] = esp_filter

    # if ESP_GRAD_KEY is not None and dft_e_field.size > 0:
    #     e_field_filter = np.any(np.abs(dft_e_field) > 1.8, axis=1)
    #     molecule_df["High ESP Gradient Magnitude"] = e_field_filter


    force_labels = [f"{args.labels[0]} Force", f"{args.labels[1]} Force"]
    molecule_indices = np.repeat(np.arange(dft_forces.shape[0]), dft_forces.shape[1])
    atom_indices = np.tile(np.repeat(np.arange(dft_forces.shape[1]//3), 3), dft_forces.shape[0])

    atom_df = pd.DataFrame({
        force_labels[0]: dft_forces_flat,
        force_labels[1]: dftb_forces_flat,
        "Atomic Number": np.repeat(dft_atomic_numbers.flatten(), 3),
        "Element": np.repeat(dft_elements.flatten(), 3),
        "Molecule Index": molecule_indices,
        "Atom Index": atom_indices,
        "Coordinate": np.repeat(["x", "y", "z"], dft_forces.shape[0] * (dft_forces.shape[1] // 3)),
    })

    # Convert Element to categorical ordered by atomic number
    unique_elements = atom_df['Element'].unique()
    ordered_elements = sorted(unique_elements, key=lambda x: atomic_numbers[x])
    atom_df['Element'] = pd.Categorical(atom_df['Element'], categories=ordered_elements, ordered=True)

    if args.source is not None:
        atom_df['Data Source'] = molecule_df['Data Source'].repeat(dft_forces.shape[1]).reset_index(drop=True)
        #atom_df = atom_df[atom_df['Data Source'].isin(["300K Simulation", "500K Simulation", "Halved H-bond Constant Simulation"])]
    # if ESP_KEY is not None and dft_esps.size > 0:
    #     atom_df["ESP"] = np.repeat(dft_esps.flatten(), 3)
    # if ESP_GRAD_KEY is not None and dft_e_field.size > 0:
    #     atom_df["ESP Gradient Magnitude"] = np.repeat(dft_e_field.flatten(), 3)

    print("Plotting correlations...")
    sns.set_context("talk", font_scale=1.3)
    sns.set_style("whitegrid")
    plot_correlation(molecule_df, energy_stats, energy_labels, "energy", ENERGY_UNIT, args)
    if ESP_KEY is not None:
        print("Plotting correlations with ESP hue...")
        plot_correlation(
            molecule_df, 
            energy_stats, 
            energy_labels, 
            "energy_esp",
            ENERGY_UNIT, 
            args,
            hue="High ESP Magnitude"
        )
    if ESP_GRAD_KEY is not None:
        print("Plotting correlations with ESP Gradient hue...")
        plot_correlation(
            molecule_df, 
            energy_stats, 
            energy_labels, 
            "energy_esp_gradient",
            ENERGY_UNIT, 
            args,
            hue="High ESP Gradient Magnitude"
        )

    plot_correlation(atom_df, force_stats, force_labels, "force", FORCE_UNIT, args)
    if ESP_KEY is not None:
        print("Plotting force correlations with ESP hue...")
        plot_correlation(
            atom_df, 
            force_stats, 
            force_labels, 
            "force_esp",
            FORCE_UNIT, 
            args,
            color="ESP"
        )
    if ESP_GRAD_KEY is not None:
        print("Plotting force correlations with ESP Gradient hue...")
        plot_correlation(
            atom_df, 
            force_stats, 
            force_labels, 
            "force_esp_gradient",
            FORCE_UNIT, 
            args,
            color="ESP Gradient Magnitude"
        )

    plot_subplots_per_element(
        atom_df,
        force_labels,
        args
    )
    plot_subplots_per_element(
        atom_df,
        force_labels,
        args,
        hue="Atom Index"
    )
    if args.source is not None:
        plot_subplots_per_element(
            atom_df,
            force_labels,
            args,
            hue="Data Source"
        )

if __name__ == '__main__':
    main()