#!/usr/bin/env python3
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker

FILEPATH = "parallelization_timings.csv"
OUTPUT_PLOT = "parallelization_scaling.png"

FIG_SIZE = (16 / 1.5, 9 / 1.5)
DPI = 150


def prepare_data(filepath: str) -> pd.DataFrame:
    """Load and prepare parallelization timing data from CSV file."""
    df = pd.read_csv(filepath)
    # Calculate mean and std for each number of walkers and device
    df_grouped = df.groupby(["Run", "Device"]).agg(
        ns_per_day_mean=("ns_per_day", "mean"),
        ns_per_day_std=("ns_per_day", "std"),
        hour_per_ns_mean=("hour_per_ns", "mean"),
        hour_per_ns_std=("hour_per_ns", "std"),
    ).reset_index()
    return df_grouped


def create_plot(df: pd.DataFrame, output_file: str) -> None:
    """Create and save parallelization scaling plot."""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE, dpi=DPI)

    # Get color palette
    colors = sns.color_palette("tab10")
    device_colors = {"GPU": colors[0], "CPU": colors[1]}
    
    # Plot each device separately
    for i, (device, device_df) in enumerate(df.groupby("Device")):
        ax.errorbar(
            device_df["Run"],
            device_df["ns_per_day_mean"],
            yerr=device_df["ns_per_day_std"],
            marker="o",
            markersize=8,
            linewidth=2,
            capsize=5,
            capthick=2,
            color=device_colors[device],
            label=device,
        )
    
    ax.set_xlabel("Number of Walkers")
    ax.set_ylabel("Performance (ns/day)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(df["Run"].unique())
    ax.set_xticklabels(df["Run"].unique())
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=DPI)
    plt.close(fig)


def main() -> None:
    """Main function to generate parallelization scaling plot."""
    sns.set_context("talk")

    df = prepare_data(FILEPATH)
    create_plot(df, OUTPUT_PLOT)
    print(f"Plot saved to {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
