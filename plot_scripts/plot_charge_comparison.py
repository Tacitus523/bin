#!/usr/bin/env python3
import argparse
import itertools
from typing import List, Optional, Dict
import os
import warnings

from ase import Atoms
from ase.io import read
from ase.data import atomic_numbers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
import pandas as pd

# LABELS = ["Mulliken", "Löwdin", "Hirshfeld", "ESP"]
# VAC_CHARGE_FILES = [
#     "charges_mull_vac.txt",
#     "charges_loew_vac.txt",
#     "charges_hirsh_vac.txt",
#     "charges_esp_vac.txt"
# ]
# WATER_CHARGE_FILES = [
#     "charges_mull_env.txt",
#     "charges_loew_env.txt",
#     "charges_hirsh_env.txt",
#     "charges_esp_env.txt"
# ]
# ENV_LABELS = ["Vacuum", "Water", "Difference"]

LABELS = ["Mulliken vac", "Mulliken env"]
VAC_CHARGE_FILES = [
    "charges_dftb_vac.txt",
    "charges_dftb_env.txt"
]

WATER_CHARGE_FILES = [
    "charges_dft_vac.txt",
    "charges_dft_env.txt"
]
ENV_LABELS = ["DFT", "DFTB", "Difference"]

GEOMS_FILE = "geoms.extxyz"

TOTAL_CHARGE = 0 # Just for a warning if the sum of charges differs from this value
BINS = 30
ALPHA = 0.7

FIGSIZE = (16,9)
CORR_FIGSIZE = (10,8)
MAX_POINTS = 10000 # Subsample points for correlation plots
TITLE = False
DPI = 100

PALETTE = sns.color_palette("tab10")
PALETTE.pop(3) # Remove red color

# Silence seaborn UserWarning about palette length
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"The palette list has more values .* than needed .*",
)

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot charge histograms and boxplots for comparison between vacuum and water charges.")
    ap.add_argument("-v", type=str, dest="vacuum_charge_files", nargs="+", default=VAC_CHARGE_FILES, required=False, help="File(s) with vacuum charge data", metavar="vacuum charge file(s)")
    ap.add_argument("-e", type=str, dest="env_charge_files", nargs="+", default=WATER_CHARGE_FILES, required=False, help="File(s) with environment charge data (optional)", metavar="environment charge file(s)")
    ap.add_argument("-g", type=str, dest="geoms_file", required=False, default=GEOMS_FILE, help="File with geometry data", metavar="geometry file")
    ap.add_argument("-l", type=str, dest="labels", nargs="+", required=False, default=LABELS, help="Labels for the charge types", metavar="labels")
    ap.add_argument("-n", "--env_names", nargs="+", dest="env_labels", required=False, default=ENV_LABELS, help="Names for the environments (required if -e is used)", metavar="environment labels")
    ap.add_argument("-t", "--total", type=float, dest="total_charge", required=False, default=TOTAL_CHARGE, help="Total charge expected in the system", metavar="total charge")
    ap.add_argument("-ov", nargs="+", dest="other_vacuum_files", required=False, default=None, help="Additional files with charge data for comparison", metavar="other charge file(s)")
    ap.add_argument("-oe", nargs="+", dest="other_env_files", required=False, default=None, help="Additional environment files with charge data for comparison", metavar="other charge file(s)")
    ap.add_argument("-x", "--extra", required=False, default=None, help="Extra label for other charge files", metavar="extra label")
    ap.add_argument("-f", "--file", type=str, dest="output_file", required=False, default="charges_simple_boxplot.png", help="Output filename for the boxplot", metavar="output file")
    ap.add_argument("-s", "--suffix", type=str, dest="suffix", required=False, default="", help="Suffix to add to output filenames", metavar="suffix")
    ap.add_argument('--index_files', type=str, nargs="*", help='Paths to index files for selecting geometries(folder_index.txt)', default=None)
    
    args = ap.parse_args()
    
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    validate_args(args)
    return args

def validate_args(args: argparse.Namespace) -> None:
    for file in args.vacuum_charge_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Charge file '{file}' does not exist.")
    if len(args.vacuum_charge_files) != len(args.labels):
        raise ValueError(f"Number of charge files must match number of labels. Got {len(args.vacuum_charge_files)} files and {len(args.labels)} labels.")
    
    # Generate default labels if not provided
    if args.labels is None:
        args.labels = [f"Charge_{i+1}" for i in range(len(args.vacuum_charge_files))]

    if args.other_vacuum_files is not None:
        for file in args.other_vacuum_files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"Other charge file '{file}' does not exist.")
        if args.extra is None:
            args.extra = "Other"

    # Check if environment files are provided
    if args.env_charge_files is not None:
        # Full comparison mode with environments
        if len(args.vacuum_charge_files) != len(args.labels):
            raise ValueError("Number of vacuum charge files must match number of labels.")
        if len(args.env_charge_files) != len(args.labels):
            raise ValueError("Number of environment charge files must match number of labels.")
        for file in args.vacuum_charge_files + args.env_charge_files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"Charge file '{file}' does not exist.")
        if len(args.env_labels) != 3:
            raise ValueError("There must be exactly three environment labels (e.g., Vacuum, Water, Difference).")
        if args.other_vacuum_files is not None:
            if args.other_env_files is None:
                raise ValueError("If other vacuum files are provided, other environment files must also be provided.")
            if len(args.other_vacuum_files) != len(args.other_env_files):
                raise ValueError("Number of other vacuum files must match number of other environment files.")
            for file in args.other_vacuum_files + args.other_env_files:
                if not os.path.exists(file):
                    raise FileNotFoundError(f"Other charge file '{file}' does not exist.")

    if args.index_files is not None:
        target = 0
        if args.vacuum_charge_files is not None:
            target += len(args.vacuum_charge_files)
        if args.env_charge_files is not None:
            target += len(args.env_charge_files)
        if args.other_vacuum_files is not None:
            target += len(args.other_vacuum_files)
        if args.other_env_files is not None:
            target += len(args.other_env_files)
        if len(args.index_files) != target:
            raise ValueError("If index files are provided, there must be exactly one file per charge file (vacuum files and environment files).")
        else:
            for file in args.index_files:
                if not os.path.exists(file):
                    raise FileNotFoundError(f"Index file '{file}' does not exist.")

    if not os.path.exists(args.geoms_file):
        raise FileNotFoundError(f"Geometry file '{args.geoms_file}' does not exist.")
    return

def prepare_data(args: argparse.Namespace) -> pd.DataFrame:
    indices = None # Used for filtering geometries to those where all methods converged
    if args.index_files is not None:
        print("Applying index files to filter geometries...")
        indices = compare_folder_orders(args.index_files)

    # Load vacuum charges and geometry
    vacuum_charges = [np.loadtxt(file) for file in args.vacuum_charge_files]
    if args.index_files is not None:
        vacuum_charges = [data[indices[idx]] for idx, data in enumerate(vacuum_charges)]
    elements = read_geoms(args.geoms_file, indices=indices[0] if args.index_files is not None else None)
    charge_labels = args.labels

    # Validate shapes
    for data in vacuum_charges:
        assert data.shape == elements.shape, f"Data shape {data.shape} does not match elements shape {elements.shape}"

    environmental_charges = None
    env_labels = None
    if args.env_charge_files is not None:
        print("Running full comparison mode with environment data...")

        environmental_charges = [np.loadtxt(file) for file in args.env_charge_files]
        if args.index_files is not None:
            index_offset = len(args.vacuum_charge_files)
            environmental_charges = [data[indices[idx + index_offset]] for idx, data in enumerate(environmental_charges)]
        env_labels = args.env_labels

        for data in environmental_charges:
            assert data.shape == elements.shape, f"Data shape {data.shape} does not match elements shape {elements.shape}"

    print("Constructing data frame...")
    dataframe = construct_dataframe(vacuum_charges, elements, charge_labels, env_charges_list=environmental_charges, env_labels=env_labels)

    if args.other_vacuum_files is not None:
        other_vacuum_charges = [np.loadtxt(file) for file in args.other_vacuum_files]
        if args.index_files is not None:
            index_offset = len(args.vacuum_charge_files) + (len(args.env_charge_files) if args.env_charge_files is not None else 0)
            other_vacuum_charges = [data[indices[idx + index_offset]] for idx, data in enumerate(other_vacuum_charges)]
        other_charge_labels = [args.extra]*len(other_vacuum_charges)
        for data in other_vacuum_charges:
            assert data.shape == elements.shape, f"Data shape {data.shape} does not match elements shape {elements.shape}"
        vacuum_charges.extend(other_vacuum_charges)
        charge_labels.extend([args.extra]*len(other_vacuum_charges))

        other_environmental_charges = None
        if args.other_env_files is not None:
            other_environmental_charges = [np.loadtxt(file) for file in args.other_env_files]
            if args.index_files is not None:
                index_offset += len(args.other_vacuum_files)
                other_environmental_charges = [data[indices[idx + index_offset]] for idx, data in enumerate(other_environmental_charges)]
            for data in other_environmental_charges:
                assert data.shape == elements.shape, f"Data shape {data.shape} does not match elements shape {elements.shape}"
            environmental_charges.extend(other_environmental_charges)

        other_dataframe = construct_dataframe(other_vacuum_charges, elements, other_charge_labels, env_charges_list=other_environmental_charges, env_labels=env_labels)
        dataframe = pd.concat([dataframe, other_dataframe], ignore_index=True)


    for i, charges in enumerate(vacuum_charges):
        total_charges = np.sum(charges, axis=1)
        if not np.allclose(total_charges, args.total_charge, atol=2e-2):
            print(f"WARNING: Total charge for '{charge_labels[i]}' in vacuum does not match expected value {args.total_charge}. Found: {total_charges}")
    if environmental_charges is not None:
        for i, charges in enumerate(environmental_charges):
            total_charges = np.sum(charges, axis=1)
            if not np.allclose(total_charges, args.total_charge, atol=2e-2):
                print(f"WARNING: Total charge for '{charge_labels[i]}' in environment does not match expected value {args.total_charge}. Found: {total_charges}")

    return dataframe

def read_geoms(file: str, indices: Optional[np.ndarray] = None) -> np.ndarray:
    """Read geometries from extxyz file and return array of element symbols."""
    molecules: List[Atoms] = read(file, index=":")

    elements = []
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
                raise ValueError(f"Index {indices[next_index]} not found in file {file}")
        elements.append(molecule.get_chemical_symbols())
    
    elements = np.array(elements, dtype=object)
    return elements

def compare_folder_orders(
        folder_order_files: Optional[List[str]]
    ) -> Optional[List[np.ndarray]]:
    """Compare folder order files and return lists of shared indices."""

    if folder_order_files is None:
        return None

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

def construct_dataframe(
    charges_list: List[np.ndarray], 
    elements: np.ndarray, 
    labels: List[str],
    env_charges_list: Optional[List[np.ndarray]] = None,
    env_labels: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Construct a dataframe from charge data.
    
    Args:
        charges_list: List of charge arrays, each with shape (n_molecules, n_atoms)
        elements: Array of element symbols with shape (n_molecules, n_atoms)
        labels: List of labels for each charge type
        env_charges_list: Optional list of environment charge arrays for comparison mode
        env_labels: Optional list of environment labels (defaults to global ENV_LABELS)
        
    Returns:
        DataFrame with charge data, elements, charge types, and optionally environments
    """
    if env_charges_list is not None:
        # Full comparison mode with environments
        n_envs = len(env_labels)
        n_charge_types = len(labels)
        
        diffs = [(env_charges_list[i] - charges_list[i]) for i in range(n_charge_types)]
        charge_type_data_list = [charges_list, env_charges_list, diffs]
        
        data_frames = []
        for env_idx in range(n_envs):
            for charge_type_idx in range(n_charge_types):
                env_label = env_labels[env_idx]
                charge_type_label = labels[charge_type_idx]
                charge_type_data = charge_type_data_list[env_idx][charge_type_idx]
                n_molecules = charge_type_data.shape[0]
                n_atoms = charge_type_data.shape[1]
                df = pd.DataFrame({
                    "Charge": charge_type_data.flatten(),
                    "Charge type": charge_type_label,
                    "Element": elements.flatten(),
                    "Atomic number": [atomic_numbers[el] for el in elements.flatten()],
                    "Environment": env_label,
                    "Molecule idx": np.repeat(np.arange(n_molecules), n_atoms),
                    "Atom idx": np.tile(np.arange(n_atoms), n_molecules)
                })
                df = df.sort_values(["Charge type", "Environment", "Molecule idx", "Atomic number"])
                data_frames.append(df)
    else:
        # Simple mode without environment comparison
        data_frames = []
        for charges, label in zip(charges_list, labels):
            n_molecules = charges.shape[0]
            n_atoms = charges.shape[1]
            
            df = pd.DataFrame({
                "Charge": charges.flatten(),
                "Element": elements.flatten(),
                "Atomic number": [atomic_numbers[el] for el in elements.flatten()],
                "Charge type": label,
                "Molecule idx": np.repeat(np.arange(n_molecules), n_atoms),
                "Atom idx": np.tile(np.arange(n_atoms), n_molecules)
            })
            df = df.sort_values(["Charge type", "Molecule idx", "Atomic number"])
            data_frames.append(df)
    
    # Concatenate all dataframes into one
    combined_df = pd.concat(data_frames, ignore_index=True)

    # # Molecule with maximal charge on an atom, for figuring out outliers
    # if env_charges_list is not None:
    #     env_data = combined_df[combined_df["Environment"].isin(env_labels[:2])]  # Drop Difference environment
    #     max_charge_entry = env_data.loc[env_data["Charge"].idxmax()]
    #     min_charge_entry = env_data.loc[env_data["Charge"].idxmin()]
    #     print(f"Maximum charge entry: {max_charge_entry}, Charge: {max_charge_entry['Charge']:.2f}") 
    #     print(f"Minimum charge entry: {min_charge_entry}, Charge: {min_charge_entry['Charge']:.2f}")
    
    return combined_df

def plot_histogram(data: pd.DataFrame, env_labels: List[str]):
    charge_labels = data["Charge type"].unique()

    overall_means = data.groupby(["Environment", "Charge type", "Element"]).mean(numeric_only=True).reset_index()
    overall_stds = data.groupby(["Environment", "Charge type", "Element"]).std(numeric_only=True).reset_index()

    for label in charge_labels:
        subset = data[data["Charge type"] == label]

        fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, sharey=True)
        if TITLE:
            fig.suptitle(label + " Charge")

        n_elements = subset["Element"].nunique()
        current_axis = axes[0]
        current_env = env_labels[0]  # First environment (e.g., Vacuum)
        g = sns.histplot(
            data=subset[subset["Environment"] == current_env],
            x="Charge",
            hue="Element", 
            bins=BINS, 
            binrange=[-1.0, 0.5], 
            alpha=ALPHA, 
            ax=current_axis, 
            stat="probability", 
            common_norm=False,
            palette=PALETTE
            )

        current_axis.set_title(f"$Q_{{{env_labels[0]}}}$")
        current_axis.set_xlabel("Charge (e)")
        current_axis.set_ylabel("Probability")
        means = overall_means[(overall_means["Environment"] == current_env) & (overall_means["Charge type"] == label)]
        stds = overall_stds[(overall_stds["Environment"] == current_env) & (overall_stds["Charge type"] == label)]
        labels = [f"{means['Element'].iloc[i]: >2}: {means['Charge'].iloc[i]:5.2f}\u00B1{stds['Charge'].iloc[i]:.2f}" for i in range(len(means))]
        [text.set_text(label) for text, label in zip(g.axes.get_legend().texts, labels)]

        current_axis = axes[1]
        current_env = env_labels[1]  # Second environment (e.g., Water)
        g = sns.histplot(
            data=subset[subset["Environment"] == current_env], 
            x="Charge",
            hue="Element", 
            bins=BINS, 
            binrange=[-1.0, 0.5], 
            alpha=ALPHA, 
            ax=current_axis, 
            stat="probability", 
            common_norm=False,
            palette=PALETTE
        )

        current_axis.set_title(f"$Q_{{{env_labels[1]}}}$")
        current_axis.set_xlabel("Charge (e)")
        means = overall_means[(overall_means["Environment"] == current_env) & (overall_means["Charge type"] == label)]
        stds = overall_stds[(overall_stds["Environment"] == current_env) & (overall_stds["Charge type"] == label)]
        labels = [f"{means['Element'].iloc[i]: >2}: {means['Charge'].iloc[i]:5.2f}\u00B1{stds['Charge'].iloc[i]:.2f}" for i in range(len(means))]
        [text.set_text(label) for text, label in zip(g.axes.get_legend().texts, labels)]

        current_axis = axes[2]
        current_env = env_labels[2]  # Difference
        g = sns.histplot(
            data=subset[subset["Environment"] == current_env],
            x="Charge",
            hue="Element",
            bins=BINS, 
            binrange=[-0.25, 0.25], 
            alpha=ALPHA, 
            ax=current_axis, 
            stat="probability", 
            common_norm=False,
            palette=PALETTE
        )

        current_axis.set_title(f"$Q_{{{env_labels[1]}}} - Q_{{{env_labels[0]}}}$")
        current_axis.set_xlabel(r"$\Delta$Charge (e)")
        means = overall_means[(overall_means["Environment"] == current_env) & (overall_means["Charge type"] == label)]
        stds = overall_stds[(overall_stds["Environment"] == current_env) & (overall_stds["Charge type"] == label)]
        labels = [f"{means['Element'].iloc[i]: >2}: {means['Charge'].iloc[i]:5.2f}\u00B1{stds['Charge'].iloc[i]:.2f}" for i in range(len(means))]
        [text.set_text(label) for text, label in zip(g.axes.get_legend().texts, labels)]


        save_name = f"{'_'.join(label.split()).lower()}_charges_histogram.png"
        plt.savefig(save_name, dpi=DPI)
        plt.close()

def plot_boxplot(data):
    for charge_type in data["Charge type"].unique():
        current_data = data[data["Charge type"]==charge_type]
        #current_data = current_data[data["Environment"].isin(env_labels_original[:2])]
        n_elements = current_data["Element"].nunique()

        plt.figure(figsize=FIGSIZE)
        fig = sns.boxplot(
            data=current_data, 
            x="Environment", 
            y="Charge", 
            hue="Element", 
            #order=env_labels_original,
            palette=PALETTE,
            showfliers=False
        )
        if TITLE:
            fig.axes.set_title("Charges boxplot")
        fig.set_xlabel("Environment")
        fig.set_ylabel("Charge (e)")
        plt.tight_layout()
        save_name = f"{'_'.join(charge_type.split()).lower()}_charges_boxplot.png"
        plt.savefig(save_name, dpi=DPI)

def plot_simple_boxplot(data_frame: pd.DataFrame, y_label: str = "Charge (e)", save_name="charges_boxplot.png") -> None:
    """
    Create a simple boxplot for multiple charge types without environment comparison.
    
    Args:
        charges_list: List of charge arrays, each with shape (n_molecules, n_atoms)
        elements: Array of element symbols with shape (n_molecules, n_atoms)
        labels: List of labels for each charge type
    """
    n_elements = len(data_frame["Element"].unique())

    # Create boxplot
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.boxplot(
        data=data_frame, 
        hue="Element", 
        y="Charge", 
        x="Charge type",
        palette=PALETTE,
        showfliers=False,
        ax=ax,
    )
    if TITLE:
        ax.set_title("Charges Comparison")
    ax.set_xlabel("Method")
    ax.set_ylabel(y_label)

    # Grid
    ax.yaxis.grid(True)

    # # Legend outside of plot
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Element")
    sns.move_legend(ax, "upper left")
    plt.tight_layout()

    plt.savefig(save_name, dpi=DPI)
    print(f"Saved boxplot to: {save_name}")
    
    # # Print statistics
    # print("\nCharge statistics by element and charge type:")
    # stats = data_frame.groupby(["Charge type", "Element"])["Charge"].agg(["mean", "std", "min", "max"])
    # pd.set_option("display.precision", 3) # set pandas precision for better readability
    # print(stats)

def create_charge_correlation_plot(
    combined: pd.DataFrame,
    charge_type: str,
    env_labels: List[str],
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Create a charge correlation scatter plot.
    
    Args:
        combined: DataFrame with combined charge data
        charge_type: Type of charge being plotted
        env_labels: List of environment labels
        ax: Optional axes to plot on (if None, creates new figure and saves it)
        
    Returns:
        The axes object with the plot
    """
    # Create figure if needed
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=CORR_FIGSIZE)

    n_elements = combined["Element"].nunique()
    # Create the scatter plot
    sns.scatterplot(
        data=combined, 
        x=env_labels[0], 
        y=env_labels[1], 
        hue="Element",
        palette=PALETTE,
        alpha=ALPHA,
        s=10,
        ax=ax
    )

    ax.legend(markerscale=3, frameon=True, title="Element", loc="lower right")
    for legend_handle in ax.get_legend().legend_handles:
        legend_handle.set_alpha(1)
        if hasattr(legend_handle, "set_sizes"):
            legend_handle.set_sizes([30])

    # Add perfect correlation line
    max_val = max(combined[env_labels[0]].max(), combined[env_labels[1]].max())
    min_val = min(combined[env_labels[0]].min(), combined[env_labels[1]].min())
    ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")

    ax.set_xlabel(f"{env_labels[0]} Charge (e)")
    ax.set_ylabel(f"{env_labels[1]} Charge (e)")

    ax.grid(True, alpha=0.3)

    # Calculate and display statistics
    stats = calculate_statistics(data1=combined[env_labels[0]], data2=combined[env_labels[1]])
    # Add annotation with statistics
    annotation_text = (
        f'RMSE: {stats["rmse"]:.2f} e\n'
        f'R²: {stats["r_squared"]:.3f}'
    )
    
    ax.annotate(
        annotation_text,
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=1, alpha=0.7),
        va='top'
    )

    # Save individual plot if it's a standalone
    if standalone:
        plt.tight_layout()
        save_name = f"correlation_charge_{'_'.join(charge_type.split()).lower()}.png"
        plt.savefig(save_name, dpi=DPI)
        plt.close()
    else:
        ax_title = f"{charge_type.replace('_', ' ').title()}" 
        ax.set_title(ax_title)

    return ax

def plot_correlation(data: pd.DataFrame, env_labels: List[str]) -> None:
    """
    Plot correlation between different charge types.
    
    Args:
        data: DataFrame with charge data
    """
    charge_types = data["Charge type"].unique()
    n_charge_types = len(charge_types)
    n_cols = 3
    n_rows = (n_charge_types + n_cols - 1) // n_cols  # Round up to the nearest whole number

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(CORR_FIGSIZE[0] * n_cols, CORR_FIGSIZE[1] * n_rows))
    axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing
    if TITLE:
        fig.suptitle("Charge Correlation")
    
    for i, charge_type in enumerate(charge_types):
        charge_type_subset = data[data["Charge type"] == charge_type]
        
        # Extract data for the specific charge types
        subset_1 = charge_type_subset[charge_type_subset["Environment"] == env_labels[0]].copy()
        subset_2 = charge_type_subset[charge_type_subset["Environment"] == env_labels[1]].copy()
        if subset_1.empty or subset_2.empty:
            print(f"WARNING: No data available for charge type '{charge_type}' in environment '{env_labels[0]}' or '{env_labels[1]}'. Skipping this charge type.")
            continue

        subset_1["global_id"] = subset_1.index
        subset_2["global_id"] = subset_2.index
        
        # Create pivot tables with frame_atom_id as index and Element as additional column
        pivot_1 = subset_1.pivot_table(index=["global_id", "Element", "Atomic number"], values="Charge").reset_index()#.sort_values("global_id").reset_index()
        pivot_2 = subset_2.pivot_table(index=["global_id", "Element", "Atomic number"], values="Charge").reset_index()#.sort_values("global_id").reset_index()

        # Combine the pivot tables
        combined = pd.DataFrame({
            env_labels[0]: pivot_1["Charge"],
            env_labels[1]: pivot_2["Charge"],
            "Element": pivot_1["Element"],
            "Atomic number": pivot_1["Atomic number"]
        })

        if combined.empty:
            print(f"WARNING: No data available for charge type '{charge_type}' in environment '{env_labels[0]}' or '{env_labels[1]}'. Skipping this charge type.")
            continue

        # Subsample if too many points
        combined = combined.sample(n=min(len(combined), MAX_POINTS), random_state=42)
        combined = combined.sort_values(by="Atomic number")

        # Create plot on the subplot axes
        create_charge_correlation_plot(
            combined=combined,
            charge_type=charge_type,
            env_labels=env_labels,
            ax=axes[i]
        )
        
        # Also create individual plot
        create_charge_correlation_plot(
            combined=combined,
            charge_type=charge_type,
            env_labels=env_labels
        )

    # Hide any unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
        
    plt.subplots_adjust(top=0.9)  # Adjust the top to make space for the suptitle
    plt.tight_layout()
    plt.savefig(f"correlation_charge.png", dpi=DPI)
    plt.close()

    # Plot difference correlations
    n_charge_combinations = n_charge_types * (n_charge_types - 1) // 2
    n_rows = (n_charge_combinations + n_cols - 1) // n_cols  # Round up to the nearest whole number

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(CORR_FIGSIZE[0] * n_cols, CORR_FIGSIZE[1] * n_rows))
    axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing
    if TITLE:
        fig.suptitle("Charge Difference Correlation")

    for i, (charge_type_1, charge_type_2) in enumerate(itertools.combinations(charge_types, 2)):
        subset_1 = data[(data["Charge type"] == charge_type_1) & (data["Environment"] == "Difference")].copy()
        subset_2 = data[(data["Charge type"] == charge_type_2) & (data["Environment"] == "Difference")].copy()
        if subset_1.empty or subset_2.empty:
            print(f"WARNING: No data available for charge types '{charge_type_1}' or '{charge_type_2}' in 'Difference' environment. Skipping this pair.")
            continue

        subset_1["global_id"] = subset_1.index
        subset_2["global_id"] = subset_2.index

        # Create pivot tables with frame_atom_id as index and Element as additional column
        pivot_1 = subset_1.pivot_table(index=["global_id", "Element", "Atomic number"], values="Charge").reset_index()#.sort_values("global_id").reset_index()
        pivot_2 = subset_2.pivot_table(index=["global_id", "Element", "Atomic number"], values="Charge").reset_index()#.sort_values("global_id").reset_index()

        # Combine the pivot tables
        combined = pd.DataFrame({
            charge_type_1: pivot_1["Charge"],
            charge_type_2: pivot_2["Charge"],
            "Element": pivot_1["Element"],
            "Atomic number": pivot_1["Atomic number"]
        })

        if combined.empty:
            print(f"WARNING: No data available for charge types '{charge_type_1}' or '{charge_type_2}' in 'Difference' environment. Skipping this pair.")
            continue

        # Subsample if too many points
        combined = combined.sample(n=min(len(combined), MAX_POINTS), random_state=42)
        combined = combined.sort_values(by="Atomic number")

        # Create plot on the subplot axes
        create_charge_correlation_plot(
            combined=combined,
            charge_type=f"{charge_type_1}_vs_{charge_type_2}",
            env_labels=[charge_type_1, charge_type_2],
            ax=axes[i]
        )
        
    # Hide any unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.subplots_adjust(top=0.9)  # Adjust the top to make space for the suptitle
    plt.tight_layout()
    plt.savefig(f"correlation_charge_difference.png", dpi=DPI)
    plt.close()

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
    
    return {
        'correlation': corr,
        'rmse': rmse,
        'mae': mae,
        'r_squared': r_squared
    }

def main() -> None:
    args = parse_args()
    sns.set_context("talk", font_scale=1.3)

    dataframe = prepare_data(args)

    if args.env_charge_files is None:
        print("Plotting simple boxplot...")
        save_name = f"charges_simple_boxplot{args.suffix}.png"
        plot_simple_boxplot(dataframe, save_name=save_name)

    else:
        # Create individual boxplots for each environment
        for environment in args.env_labels:
            subset = dataframe[dataframe["Environment"] == environment]
            if environment == "Difference":
                y_label = r"$\Delta$Charge (e)"
            else:
                y_label = "Charge (e)"
            save_name = f"charges_boxplot_{environment.lower()}{args.suffix}.png"
            plot_simple_boxplot(subset, y_label=y_label, save_name=save_name)

        # Remove additionally given charge files from dataframe for histograms and correlations
        if hasattr(args, "extra"):
            dataframe = dataframe[dataframe["Charge type"] != args.extra]

        print("Plotting histograms...")
        plot_histogram(dataframe, args.env_labels)
        print("Plotting boxplots...")
        plot_boxplot(dataframe)
        print("Plotting correlations...")
        plot_correlation(dataframe, args.env_labels)

if __name__=="__main__":
    main()
