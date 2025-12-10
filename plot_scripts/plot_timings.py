#!/usr/bin/env python3
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker

FILEPATH = "timing_results.csv"
OUTPUT_PLOT = "performance_comparison.png"

charge_unit = "e"
energy_unit = "eV"
force_unit = "eV/Å"

eV_to_kcal_mol = 23.0605
kcal_mol_to_eV = 1 / eV_to_kcal_mol

Hartree_to_eV = 27.2114
bohr_to_angstrom = 0.529177
hartree_bohr_to_eV_angstrom = Hartree_to_eV * 1 / bohr_to_angstrom

FIG_SIZE = (16/1.5, 9/1.5)
DPI = 150

clean_names = {
    "amber99sb-ildn": "Amber99SB-ILDN",
    "dftb": "DFTB3",
    "hdnnp2nd": "2G-HDNNP",
    "hdnnp4th": "4G-HDNNP",
    "schnet": "SchNet",
    "painn": "PaiNN",
    "base_mace": "Base MACE",
    "maceqeq": "QEq-MACE",
    "amp": "AMP",
}

def prepare_data(filepath: str) -> pd.DataFrame:
    """Load and prepare timing data from CSV file."""
    df = pd.read_csv(filepath)
    df["Method"] = df["Method"].map(clean_names).fillna(df["Method"])
    df["Method"] = pd.Categorical(df["Method"], categories=clean_names.values(), ordered=True)
    return df


def create_plot(df: pd.DataFrame, output_file: str) -> None:
    """Create and save performance comparison bar plot."""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    
    # Create bar plot
    sns.barplot(x="Method", y="ns_per_day", hue="Device", data=df, ax=ax, palette="tab10")
    
    # Configure axes
    ax.set_yscale("log")
    ax.set_ylabel("Performance (ns/day)")
    ax.set_xlabel("Method")
    
    # Add annotations to each bar
    for bar in ax.patches:
        height = bar.get_height()
        if np.isnan(height):
            continue
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    
    # Configure axis styling
    ax.yaxis.set_minor_locator(
        ticker.LogLocator(base=10.0, subs=[2, 3, 4, 5, 6, 7, 8, 9])
    )
    ax.tick_params(axis="y", which="minor", length=4, width=0.8)
    ax.grid(True, which="minor", alpha=0.3)
    ax.set_ylim(bottom=0.1, top=30)
    
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_file, dpi=DPI)
    plt.close(fig)


def main() -> None:
    """Main function to generate performance comparison plot."""
    sns.set_context("talk")
    
    df = prepare_data(FILEPATH)
    create_plot(df, OUTPUT_PLOT)


if __name__ == "__main__":
    main()


