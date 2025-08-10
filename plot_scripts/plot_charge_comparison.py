#!/usr/bin/env python3
import argparse
import itertools
from typing import List, Optional
import os

from ase import Atoms
from ase.io import read
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    "charges_dft_vac.txt"
    "charges_dft_env.txt"
]
ENV_LABELS = ["DFT", "DFTB", "Difference"]

GEOMS_FILE = "geoms_vac.extxyz"

TOTAL_CHARGE = 0 # Just for a warning if the sum of charges differs from this value
BINS = 30
ALPHA = 0.7

FIGSIZE = (8, 6)
FONTSIZE = 24
LABELSIZE = 20
TITLE = False
DPI = 100

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot charge histograms and boxplots for comparison between vacuum and water charges.")
    ap.add_argument("-v", type=str, dest="vacuum_charge_files", nargs="+", default=VAC_CHARGE_FILES, required=False, help="File(s) with vacuum charge data", metavar="vacuum charge file(s)")
    ap.add_argument("-e", type=str, dest="env_charge_files", nargs="+", default=WATER_CHARGE_FILES, required=False, help="File(s) with environment charge data", metavar="environment charge file(s)")
    ap.add_argument("-g", type=str, dest="geoms_file", required=False, default=GEOMS_FILE, help="File with geometry data", metavar="geometry file")
    ap.add_argument("-l", type=str, dest="labels", nargs="+", required=False, default=LABELS, help="Labels for the charge types", metavar="labels")
    ap.add_argument("-t", "--total", type=float, dest="total_charge", required=False, default=TOTAL_CHARGE, help="Total charge expected in the system", metavar="total charge")
    args = ap.parse_args()
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    validate_args(args)
    return args

def validate_args(args: argparse.Namespace) -> None:
    if len(args.vacuum_charge_files) != len(args.labels):
        raise ValueError("Number of vacuum charge files must match number of labels.")
    if len(args.env_charge_files) != len(args.labels):
        raise ValueError("Number of environment charge files must match number of labels.")
    if not os.path.exists(args.geoms_file):
        raise FileNotFoundError(f"Geometry file '{args.geoms_file}' does not exist.")
    if not isinstance(args.labels, list):
        raise TypeError("Labels must be provided as a list.")
    if not isinstance(args.vacuum_charge_files, list):
        raise TypeError("Vacuum charge files must be provided as a list.")
    if not isinstance(args.env_charge_files, list):
        raise TypeError("Environment charge files must be provided as a list.")
    for file in args.vacuum_charge_files + args.env_charge_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Charge file '{file}' does not exist.")
    return

def read_geoms(file: str) -> np.ndarray:
    geoms: List[Atoms] = read(file, index=":")

    elements = []
    for geom in geoms:
        elements.append(geom.get_chemical_symbols())
    
    elements = np.array(elements, dtype=object)
    return elements

def construct_dataframe(vacs: List[np.ndarray], waters: List[np.ndarray], charge_type_labels: List[str], elements: np.ndarray) -> pd.DataFrame:
    n_envs = len(ENV_LABELS)
    n_charge_types = len(charge_type_labels)

    diffs = [(waters[i] - vacs[i]) for i in range(n_charge_types)]
    charge_type_data_list = [vacs, waters, diffs]

    data_frames = []
    for env_idx in range(n_envs):
        for charge_type_idx in range(n_charge_types):
            env_label = ENV_LABELS[env_idx] # shape: (1,)
            charge_type_label = charge_type_labels[charge_type_idx] # shape: (1,)
            charge_type_data = charge_type_data_list[env_idx][charge_type_idx] # shape: (n_molecules, n_atoms)
            n_molecules = charge_type_data.shape[0]
            n_atoms = charge_type_data.shape[1]
            df = pd.DataFrame({
                "Charge": charge_type_data.flatten(),
                "Charge type": charge_type_label,
                "Element": elements.flatten(),
                "Environment": env_label,
                "Dataset molecule idx": np.repeat(np.arange(n_molecules), n_atoms),
                "Dataset atom idx": np.tile(np.arange(n_atoms), n_molecules)
            })
            data_frames.append(df)
    # Concatenate all dataframes into one
    data = pd.concat(data_frames, ignore_index=True)

    # Molecule with maximal charge on an atom
    env_data = data[data["Environment"].isin(ENV_LABELS[:2])]  # Drop Difference environment
    max_charge_entry = env_data.loc[env_data["Charge"].idxmax()]
    min_charge_entry = env_data.loc[env_data["Charge"].idxmin()]
    print(f"Maximum charge entry: {max_charge_entry}, Charge: {max_charge_entry['Charge']:.2f}") 
    print(f"Minimum charge entry: {min_charge_entry}, Charge: {min_charge_entry['Charge']:.2f}")
    return data

def plot_histogram(data: pd.DataFrame):
    charge_labels = data["Charge type"].unique()

    overall_means = data.groupby(["Environment", "Charge type", "Element"]).mean(numeric_only=True).reset_index().sort_values("Element")
    overall_stds = data.groupby(["Environment", "Charge type", "Element"]).std(numeric_only=True).reset_index().sort_values("Element")

    for label in charge_labels:
        subset = data[data["Charge type"] == label].sort_values("Element")

        fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, sharey=True)
        if TITLE:
            fig.suptitle(label + " Charge", fontsize=FONTSIZE)

        current_axis = axes[0]
        current_env = ENV_LABELS[0]  # Vacuum
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
            palette="tab10"
            )

        current_axis.set_title(r"$Q_{Vacuum}$", fontsize=LABELSIZE)
        current_axis.set_xlabel("Charge [e]", fontsize=LABELSIZE)
        current_axis.set_ylabel("Probability", fontsize=LABELSIZE)
        means = overall_means[(overall_means["Environment"] == current_env) & (overall_means["Charge type"] == label)]
        stds = overall_stds[(overall_stds["Environment"] == current_env) & (overall_stds["Charge type"] == label)]
        labels = [f"{means['Element'].iloc[i]: >2}: {means['Charge'].iloc[i]:5.2f}\u00B1{stds['Charge'].iloc[i]:.2f}" for i in range(len(means))]
        [text.set_text(label) for text, label in zip(g.axes.get_legend().texts, labels)]
        current_axis.tick_params(labelsize=LABELSIZE)
        plt.setp(current_axis.get_legend().get_texts(), fontsize=LABELSIZE)
        plt.setp(current_axis.get_legend().get_title(), fontsize=LABELSIZE)

        current_axis = axes[1]
        current_env = ENV_LABELS[1]  # Water
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
            palette="tab10"
        )

        current_axis.set_title(r"$Q_{Water}$", fontsize=LABELSIZE)
        current_axis.set_xlabel("Charge [e]", fontsize=LABELSIZE)
        means = overall_means[(overall_means["Environment"] == current_env) & (overall_means["Charge type"] == label)]
        stds = overall_stds[(overall_stds["Environment"] == current_env) & (overall_stds["Charge type"] == label)]
        labels = [f"{means['Element'].iloc[i]: >2}: {means['Charge'].iloc[i]:5.2f}\u00B1{stds['Charge'].iloc[i]:.2f}" for i in range(len(means))]
        [text.set_text(label) for text, label in zip(g.axes.get_legend().texts, labels)]
        current_axis.tick_params(labelsize=LABELSIZE)
        plt.setp(current_axis.get_legend().get_texts(), fontsize=LABELSIZE)
        plt.setp(current_axis.get_legend().get_title(), fontsize=LABELSIZE)

        current_axis = axes[2]
        current_env = ENV_LABELS[2]  # Difference
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
            palette="tab10"
        )

        current_axis.set_title(r"$Q_{Water} - Q_{Vacuum}$", fontsize=LABELSIZE)
        current_axis.set_xlabel(r"$\Delta$Charge [e]", fontsize=LABELSIZE)
        means = overall_means[(overall_means["Environment"] == current_env) & (overall_means["Charge type"] == label)]
        stds = overall_stds[(overall_stds["Environment"] == current_env) & (overall_stds["Charge type"] == label)]
        labels = [f"{means['Element'].iloc[i]: >2}: {means['Charge'].iloc[i]:5.2f}\u00B1{stds['Charge'].iloc[i]:.2f}" for i in range(len(means))]
        [text.set_text(label) for text, label in zip(g.axes.get_legend().texts, labels)]
        current_axis.tick_params(labelsize=LABELSIZE)
        plt.setp(current_axis.get_legend().get_texts(), fontsize=LABELSIZE)
        plt.setp(current_axis.get_legend().get_title(), fontsize=LABELSIZE)

        save_name = f"{'_'.join(label.split()).lower()}_charges_histogram.png"
        plt.savefig(save_name, dpi=DPI)
        plt.close()

def plot_boxplot(data):
    for charge_type in data["Charge type"].unique():
        current_data = data[data["Charge type"]==charge_type]
        #current_data = current_data[data["Environment"].isin(env_labels_original[:2])]
        plt.figure(figsize=FIGSIZE)
        fig = sns.boxplot(
            data=current_data, 
            x="Environment", 
            y="Charge", 
            hue="Element", 
            #order=env_labels_original
        )
        if TITLE:
            fig.axes.set_title("Charges boxplot", fontsize=FONTSIZE)
        fig.set_xlabel("Method", fontsize=FONTSIZE)
        fig.set_ylabel("Charge", fontsize=FONTSIZE)
        fig.tick_params(labelsize=FONTSIZE, axis="x")
        fig.tick_params(labelsize=LABELSIZE, axis="y")
        plt.setp(fig.get_legend().get_title(), fontsize=FONTSIZE) # for legend title
        plt.setp(fig.get_legend().get_texts(), fontsize=FONTSIZE) # for legend text
        plt.tight_layout()
        save_name = f"{'_'.join(charge_type.split()).lower()}_charges_boxplot.png"
        plt.savefig(save_name, dpi=DPI)

def create_charge_correlation_plot(
    combined: pd.DataFrame,
    charge_type: str,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Create a charge correlation scatter plot.
    
    Args:
        combined: DataFrame with combined charge data
        charge_type: Type of charge being plotted
        ax: Optional axes to plot on (if None, creates new figure and saves it)
        
    Returns:
        The axes object with the plot
    """
    # Create figure if needed
    standalone = ax is None
    if standalone:
        sns.set_context("talk")
        fig, ax = plt.subplots(figsize=FIGSIZE)
    
    # Create the scatter plot
    sns.scatterplot(
        data=combined, 
        x=ENV_LABELS[0], 
        y=ENV_LABELS[1], 
        hue="Element",
        palette="tab10",
        alpha=ALPHA,
        s=10,
        ax=ax
    )

    # Add perfect correlation line
    max_val = max(combined[ENV_LABELS[0]].max(), combined[ENV_LABELS[1]].max())
    min_val = min(combined[ENV_LABELS[0]].min(), combined[ENV_LABELS[1]].min())
    ax.plot([min_val, max_val], [min_val, max_val], color="k", linestyle="--", label="Perfect Correlation")

    ax.set_xlabel(f"{ENV_LABELS[0]} Charge [e]", fontsize=LABELSIZE)
    ax.set_ylabel(f"{ENV_LABELS[1]} Charge [e]", fontsize=LABELSIZE)
    ax.tick_params(labelsize=LABELSIZE)
    
    # Save individual plot if it's a standalone
    if standalone:
        plt.tight_layout()
        save_name = f"correlation_charge_{'_'.join(charge_type.split()).lower()}.png"
        plt.savefig(save_name, dpi=DPI)
        plt.close()
    else:
        # Calculate correlation coefficient
        corr = combined[ENV_LABELS[0]].corr(combined[ENV_LABELS[1]])
        ax_title = f"{charge_type.replace('_', ' ').title()} (r={corr:.3f})" 
        ax.set_title(ax_title, fontsize=LABELSIZE)
    
    return ax

def plot_correlation(data: pd.DataFrame) -> None:
    """
    Plot correlation between different charge types.
    
    Args:
        data: DataFrame with charge data
    """
    charge_types = data["Charge type"].unique()
    n_charge_types = len(charge_types)
    n_cols = 4
    n_rows = (n_charge_types + n_cols - 1) // n_cols  # Round up to the nearest whole number

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * FIGSIZE[0], n_rows * FIGSIZE[1]))
    axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing
    if TITLE:
        fig.suptitle("Charge Correlation", fontsize=FONTSIZE)
    
    for i, charge_type in enumerate(charge_types):
        charge_type_subset = data[data["Charge type"] == charge_type]
        
        # Extract data for the specific charge types
        subset_1 = charge_type_subset[charge_type_subset["Environment"] == ENV_LABELS[0]].copy()
        subset_2 = charge_type_subset[charge_type_subset["Environment"] == ENV_LABELS[1]].copy()
        if subset_1.empty or subset_2.empty:
            print(f"WARNING: No data available for charge type '{charge_type}' in environment '{ENV_LABELS[0]}' or '{ENV_LABELS[1]}'. Skipping this charge type.")
            continue

        subset_1["global_id"] = subset_1.index
        subset_2["global_id"] = subset_2.index
        
        # Create pivot tables with frame_atom_id as index and Element as additional column
        pivot_1 = subset_1.pivot_table(index=["Element", "global_id"], values="Charge").reset_index().sort_values("global_id").reset_index()
        pivot_2 = subset_2.pivot_table(index=["Element", "global_id"], values="Charge").reset_index().sort_values("global_id").reset_index()

        # Combine the pivot tables
        combined = pd.DataFrame({
            ENV_LABELS[0]: pivot_1["Charge"],
            ENV_LABELS[1]: pivot_2["Charge"],
            "Element": pivot_1["Element"],
        })

        if combined.empty:
            print(f"WARNING: No data available for charge type '{charge_type}' in environment '{ENV_LABELS[0]}' or '{ENV_LABELS[1]}'. Skipping this charge type.")
            continue

        # Create plot on the subplot axes
        create_charge_correlation_plot(
            combined=combined,
            charge_type=charge_type,
            ax=axes[i]
        )
        
        # Also create individual plot
        create_charge_correlation_plot(
            combined=combined,
            charge_type=charge_type
        )

    # Hide any unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
        
    plt.subplots_adjust(top=0.9)  # Adjust the top to make space for the suptitle
    plt.tight_layout()
    plt.savefig(f"correlation_charge.png", dpi=DPI)
    plt.close()

def main() -> None:
    args = parse_args()
    vacs = [np.loadtxt(file) for file in args.vacuum_charge_files]
    waters = [np.loadtxt(file) for file in args.env_charge_files]
    elements = read_geoms(args.geoms_file)
    charge_labels = args.labels

    assert len(vacs)==len(waters)
    assert len(vacs)==len(charge_labels)
    for data in vacs:
        assert data.shape == vacs[0].shape
    for data in waters:
        assert data.shape == vacs[0].shape
    total_charges = [np.sum(data, axis=1) for data in vacs + waters]
    for i, total_charge in enumerate(total_charges):
        if not np.allclose(total_charge, args.total_charge, atol=1e-5):
            print(f"WARNING: Total charge for input {i} does not match expected value {args.total_charge}. Found: {total_charge}")

    print("Constructing data frame...")
    data = construct_dataframe(vacs, waters, charge_labels, elements)
    sns.set_context("talk", font_scale=1.3)
    print("Plotting histograms...")
    plot_histogram(data)
    print("Plotting boxplots...")
    plot_boxplot(data)
    print("Plotting correlations...")
    plot_correlation(data)

if __name__=="__main__":
    main()
