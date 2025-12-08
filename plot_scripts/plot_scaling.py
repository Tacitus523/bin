#!/usr/bin/env python3
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker

FILEPATH = "scaling_timings.csv"
OUTPUT_PLOT = "scaling_performance.png"

FIG_SIZE = (16 / 1.5, 9 / 1.5)
DPI = 150

ATOMS_PER_MOLECULE = 22  # Number of atoms in the dipeptide system

PALETTE = sns.color_palette("tab10")
PALETTE.pop(3)  # Remove red color

# Silence seaborn UserWarning about palette length
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"The palette list has more values .* than needed .*",
)

clean_names = {
    "dftb": "DFTB3",
    "hdnnp2nd": "2G-HDNNP",
    "hdnnp4th": "4G-HDNNP",
    "schnet": "SchNet",
    "painn": "PaiNN",
    "base_mace": "Base MACE (GPU)",
    "maceqeq": "QEq-MACE (GPU)",
    #"amp": "AMP (GPU)",
}


def prepare_data(filepath: str) -> pd.DataFrame:
    """Load and prepare scaling timing data from CSV file."""
    df = pd.read_csv(filepath)
    # Clean method names
    df["Method"] = df["Method"].map(clean_names).fillna(df["Method"])
    df ["Method"] = pd.Categorical(df["Method"], categories=clean_names.values(), ordered=True)
    df["Atoms"] = df["Molecules"] * ATOMS_PER_MOLECULE
    return df


def create_plot(df: pd.DataFrame, output_file: str) -> None:
    """Create and save scaling performance plot."""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE, dpi=DPI)

    # Create line plot with seaborn
    sns.lineplot(
        data=df,
        x="Atoms",
        y="ns_per_day",
        hue="Method",
        marker="o",
        markersize=8,
        linewidth=2,
        palette=PALETTE,
        ax=ax,
    )
    
    ax.set_xlabel("Number of Atoms")
    ax.set_ylabel("Performance (ns/day)")
    ax.set_yscale("log")
    ax.set_xticks(df["Atoms"].unique())
    ax.set_xticklabels(df["Atoms"].unique())
    ax.grid(True, alpha=0.3)
    
    # Configure axis styling
    ax.yaxis.set_minor_locator(
        ticker.LogLocator(base=10.0, subs=[2, 3, 4, 5, 6, 7, 8, 9])
    )
    ax.tick_params(axis="y", which="minor", length=4, width=0.8)
    ax.grid(True, which="minor", alpha=0.3)

    ax.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(output_file, dpi=DPI)
    plt.close(fig)


def main() -> None:
    """Main function to generate scaling performance plot."""
    sns.set_context("talk")

    df = prepare_data(FILEPATH)
    create_plot(df, OUTPUT_PLOT)
    print(f"Plot saved to {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()