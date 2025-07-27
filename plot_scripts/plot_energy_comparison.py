#!/usr/bin/env python3
"""
Energy comparison plotting script for vacuum vs environment calculations.

This script loads energy data from two files, converts from Hartree to kcal/mol,
and generates various plots including histograms, boxplots, and violin plots.
"""

from typing import Tuple, List, Optional
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Default configuration values
H_TO_KCAL_MOL = 627.509

# Default file paths
DEFAULT_VACUUM_FILE = "energy_vac.txt"
DEFAULT_WATER_FILE = "energy_env.txt"

# Default plot parameters
DEFAULT_BINS = 40
DEFAULT_ALPHA = 0.5
DEFAULT_FIGSIZE = (8, 6)
DEFAULT_FONTSIZE = 20
DEFAULT_LABELSIZE = 16
DEFAULT_SHOW_TITLE = False
DEFAULT_DPI = 200
DEFAULT_OUTPUT_PREFIX = "energy"

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Compare and plot energy distributions from vacuum and environment calculations"
    )
    
    parser.add_argument(
        "vacuum_file",
        type=str,
        default=DEFAULT_VACUUM_FILE,
        help=f"Path to vacuum energy file (default: {DEFAULT_VACUUM_FILE})"
    )
    
    parser.add_argument(
        "water_file",
        type=str,
        default=DEFAULT_WATER_FILE,
        help=f"Path to water/environment energy file (default: {DEFAULT_WATER_FILE})"
    )

    parser.add_argument(
        "-s", "--sources",
        type=str,
        default=None,
        help="File with data sources for coloring, default: None"
    )
    
    parser.add_argument(
        "--bins",
        type=int,
        default=DEFAULT_BINS,
        help=f"Number of histogram bins (default: {DEFAULT_BINS})"
    )
    
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help=f"Transparency level for plots (default: {DEFAULT_ALPHA})"
    )
    
    parser.add_argument(
        "--figsize",
        type=int,
        nargs=2,
        default=list(DEFAULT_FIGSIZE),
        help=f"Figure size as width height (default: {DEFAULT_FIGSIZE[0]} {DEFAULT_FIGSIZE[1]})"
    )
    
    parser.add_argument(
        "--fontsize",
        type=int,
        default=DEFAULT_FONTSIZE,
        help=f"Font size for labels (default: {DEFAULT_FONTSIZE})"
    )
    
    parser.add_argument(
        "--labelsize",
        type=int,
        default=DEFAULT_LABELSIZE,
        help=f"Font size for tick labels (default: {DEFAULT_LABELSIZE})"
    )
    
    parser.add_argument(
        "--show-title",
        action="store_true",
        default=DEFAULT_SHOW_TITLE,
        help=f"Show titles on plots (default: {DEFAULT_SHOW_TITLE})"
    )
    
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"Plot resolution (default: {DEFAULT_DPI})"
    )
    
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=DEFAULT_OUTPUT_PREFIX,
        help=f"Prefix for output files (default: {DEFAULT_OUTPUT_PREFIX})"
    )
    
    return parser.parse_args()

def load_and_convert_energies(
    vacuum_file: str, 
    water_file: str, 
    conversion_factor: float = H_TO_KCAL_MOL
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load energy data from files and convert to kcal/mol.
    
    Args:
        vacuum_file: Path to vacuum energy file (in Hartree)
        water_file: Path to water/environment energy file (in Hartree)
        conversion_factor: Conversion factor from Hartree to kcal/mol
        
    Returns:
        Tuple of (vacuum_energies, water_energies, energy_differences)
    """
    energy_vacuum = np.loadtxt(vacuum_file) * conversion_factor
    energy_water = np.loadtxt(water_file) * conversion_factor
    energy_diff = energy_water - energy_vacuum
    
    return energy_vacuum, energy_water, energy_diff


def create_energy_dataframe(
    energy_vacuum: np.ndarray, 
    energy_water: np.ndarray, 
    energy_diff: np.ndarray,
    data_sources: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Create a pandas DataFrame from energy arrays.
    
    Args:
        energy_vacuum: Vacuum energy values
        energy_water: Water/environment energy values
        energy_diff: Energy differences
        
    Returns:
        DataFrame with energy columns
    """
    labels = [r"$E_{Vacuum}$", r"$E_{Water}$", r"$\Delta E$"]
    energies = [energy_vacuum, energy_water, energy_diff]
    
    df = pd.DataFrame()
    for energy, label in zip(energies, labels):
        df[label] = energy
    
    if data_sources is not None:
        df["Data Source"] = data_sources["Data Source"]

    return df

def create_stacked_energy_dataframe(
    energy_vacuum: np.ndarray, 
    energy_water: np.ndarray,
    data_sources: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Create a stacked DataFrame for boxplot/violinplot visualization.
    
    Args:
        energy_vacuum: Vacuum energy values
        energy_water: Water/environment energy values
        
    Returns:
        Stacked DataFrame with 'Energy' and 'Labels' columns
    """
    n_datapoints = len(energy_vacuum)
    labels = [r"$E_{Vacuum}$", r"$E_{Water}$"]
    label_arrays = [[label] * n_datapoints for label in labels]
    
    energies_stacked = np.concatenate([energy_vacuum, energy_water])
    labels_stacked = np.concatenate(label_arrays)
    
    data = pd.DataFrame()
    data["Energy"] = energies_stacked
    data["Labels"] = labels_stacked
    
    if data_sources is not None:
        data["Data Source"] = pd.concat([data_sources["Data Source"]] * len(labels)).reset_index(drop=True)

    return data

def plot_energy_histogram(
    df: pd.DataFrame, 
    output_file: str = "energy_histogram.png",
    bins: int = 40,
    alpha: float = 0.5,
    figsize: Tuple[int, int] = (16, 9),
    fontsize: int = 20,
    labelsize: int = 16,
    show_title: bool = False,
    dpi: int = 200
) -> None:
    """
    Create histogram plot for vacuum and water energies.
    
    Args:
        df: DataFrame containing energy data
        output_file: Output filename for the plot
        bins: Number of histogram bins
        alpha: Transparency level
        figsize: Figure size tuple
        fontsize: Font size for labels
        labelsize: Font size for tick labels
        show_title: Whether to show plot title
        dpi: Plot resolution
    """
    labels = [r"$E_{Vacuum}$", r"$E_{Water}$"]
    
    plt.figure(figsize=figsize)
    fig = sns.histplot(data=df[labels], bins=bins, alpha=alpha)
    
    if show_title:
        fig.axes.set_title("Energy distribution", fontsize=fontsize)
    
    fig.set_xlabel(r"Energy [$\frac{kcal}{mol}$]", fontsize=fontsize)
    fig.set_ylabel("Count", fontsize=fontsize)
    fig.tick_params(labelsize=labelsize)
    
    if fig.get_legend():
        plt.setp(fig.get_legend().get_texts(), fontsize=labelsize)
        plt.setp(fig.get_legend().get_title(), fontsize=labelsize)
    
    fig.xaxis.get_offset_text().set_fontsize(labelsize)
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi)
    plt.close()


def plot_energy_difference_histogram(
    energy_diff: np.ndarray,
    output_file: str = "energy_difference_histogram.png",
    bins: int = 40,
    alpha: float = 0.5,
    figsize: Tuple[int, int] = (16, 9),
    fontsize: int = 20,
    labelsize: int = 16,
    show_title: bool = False,
    dpi: int = 200
) -> None:
    """
    Create histogram plot for energy differences.
    
    Args:
        energy_diff: Array of energy differences
        output_file: Output filename for the plot
        bins: Number of histogram bins
        alpha: Transparency level
        figsize: Figure size tuple
        fontsize: Font size for labels
        labelsize: Font size for tick labels
        show_title: Whether to show plot title
        dpi: Plot resolution
    """
    plt.figure(figsize=figsize)
    fig = sns.histplot(data=energy_diff, bins=bins, alpha=alpha)
    
    if show_title:
        fig.axes.set_title("Energy difference distribution", fontsize=fontsize)
    
    fig.set_xlabel(r"Energy difference [$\frac{kcal}{mol}$]", fontsize=fontsize)
    fig.set_ylabel("Count", fontsize=fontsize)
    fig.tick_params(labelsize=labelsize)
    fig.xaxis.get_offset_text().set_fontsize(labelsize)
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi)
    plt.close()


def plot_energy_boxplot(
    data: pd.DataFrame,
    output_file: str = "energy_boxplot.png",
    figsize: Tuple[int, int] = (16, 9),
    fontsize: int = 20,
    labelsize: int = 16,
    show_title: bool = False,
    dpi: int = 200
) -> None:
    """
    Create boxplot for energy comparison.
    
    Args:
        data: Stacked DataFrame with energy data
        output_file: Output filename for the plot
        figsize: Figure size tuple
        fontsize: Font size for labels
        labelsize: Font size for tick labels
        show_title: Whether to show plot title
        dpi: Plot resolution
    """
    plt.figure(figsize=figsize)
    fig = sns.boxplot(
        data=data,
        x="Labels",
        y="Energy",
        hue="Data Source" if "Data Source" in data.columns else None,
        palette="tab10" if "Data Source" in data.columns else None,
    )
    
    if show_title:
        fig.axes.set_title("Energy boxplot", fontsize=fontsize)
    
    fig.set_xlabel("Energy source", fontsize=fontsize)
    fig.set_ylabel(r"Energy [$\frac{kcal}{mol}$]", fontsize=fontsize)
    fig.tick_params(labelsize=fontsize, axis="x")
    fig.tick_params(labelsize=labelsize, axis="y")
    fig.yaxis.get_offset_text().set_fontsize(labelsize)
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi)
    plt.close()


def plot_energy_violinplot(
    data: pd.DataFrame,
    output_file: str = "energy_violinplot.png",
    figsize: Tuple[int, int] = (16, 9),
    fontsize: int = 20,
    labelsize: int = 16,
    show_title: bool = False,
    dpi: int = 200
) -> None:
    """
    Create violin plot for energy comparison.
    
    Args:
        data: Stacked DataFrame with energy data
        output_file: Output filename for the plot
        figsize: Figure size tuple
        fontsize: Font size for labels
        labelsize: Font size for tick labels
        show_title: Whether to show plot title
        dpi: Plot resolution
    """
    plt.figure(figsize=figsize)
    fig = sns.violinplot(
        data=data, 
        x="Labels", 
        y="Energy", 
        hue="Data Source" if "Data Source" in data.columns else None,
        palette="tab10" if "Data Source" in data.columns else None
    )
    
    if show_title:
        fig.axes.set_title("Energy violinplot", fontsize=fontsize)
    
    fig.set_xlabel("Energy source", fontsize=fontsize)
    fig.set_ylabel(r"Energy [$\frac{kcal}{mol}$]", fontsize=fontsize)
    fig.tick_params(labelsize=fontsize, axis="x")
    fig.tick_params(labelsize=labelsize, axis="y")
    fig.yaxis.get_offset_text().set_fontsize(labelsize)
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi)
    plt.close()

def main() -> None:
    """Main function to orchestrate the energy comparison plotting."""
    args = parse_arguments()
    
    # Set seaborn context
    sns.set_context("talk")
    
    # Load and convert energy data
    energy_vacuum, energy_water, energy_diff = load_and_convert_energies(
        args.vacuum_file, args.water_file
    )

    if args.sources:
        # Load data source file if provided
        data_sources = pd.read_csv(args.sources, header=None, names=["Data Source"], sep=",")
    else:
        data_sources = None
    
    # Create DataFrames
    df = create_energy_dataframe(energy_vacuum, energy_water, energy_diff, data_sources)
    stacked_data = create_stacked_energy_dataframe(energy_vacuum, energy_water, data_sources)
    
    # Generate plots
    plot_kwargs = {
        "bins": args.bins,
        "alpha": args.alpha,
        "figsize": tuple(args.figsize),
        "fontsize": args.fontsize,
        "labelsize": args.labelsize,
        "show_title": args.show_title,
        "dpi": args.dpi
    }
    
    # Energy histogram (vacuum + water)
    plot_energy_histogram(
        df, 
        output_file=f"{args.output_prefix}_histogram.png",
        **plot_kwargs
    )
    
    # Energy difference histogram
    plot_energy_difference_histogram(
        energy_diff,
        output_file=f"{args.output_prefix}_difference_histogram.png",
        **plot_kwargs
    )
    
    # Boxplot
    plot_energy_boxplot(
        stacked_data,
        output_file=f"{args.output_prefix}_boxplot.png",
        **{k: v for k, v in plot_kwargs.items() if k != "bins" and k != "alpha"}
    )
    
    # Violin plot
    plot_energy_violinplot(
        stacked_data,
        output_file=f"{args.output_prefix}_violinplot.png",
        **{k: v for k, v in plot_kwargs.items() if k != "bins" and k != "alpha"}
    )
    
    print(f"Plots generated successfully!")
    print(f"Energy statistics:")
    print(f"  Vacuum energy: {energy_vacuum.mean():.2f} ± {energy_vacuum.std():.2f} kcal/mol")
    print(f"  Water energy: {energy_water.mean():.2f} ± {energy_water.std():.2f} kcal/mol")
    print(f"  Energy difference: {energy_diff.mean():.2f} ± {energy_diff.std():.2f} kcal/mol")


if __name__ == "__main__":
    main()
