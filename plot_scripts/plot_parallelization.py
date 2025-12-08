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
    df["Run"] = df["Run"].astype(int)
    # # Calculate mean and std for each number of walkers and device
    # df_grouped = df.groupby(["Run", "Device"]).agg(
    #     ns_per_day_mean=("ns_per_day", "mean"),
    #     ns_per_day_std=("ns_per_day", "std"),
    #     hour_per_ns_mean=("hour_per_ns", "mean"),
    #     hour_per_ns_std=("hour_per_ns", "std"),
    # ).reset_index()
    df["Device"] = pd.Categorical(df["Device"], categories=["CPU","GPU"], ordered=True)
    return df


def create_plot(df: pd.DataFrame, output_file: str) -> None:
    """Create and save parallelization scaling plot."""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE, dpi=DPI)

    # Get color palette
    palette = "tab10"
    
    # Use seaborn pointplot with hue for Device
    sns.pointplot(
        data=df,
        x="Run",
        y="ns_per_day",
        hue="Device",
        palette=palette,
        markers=["o", "s"],
        linestyles=["-", "-"],
        linewidth=2,
        capsize=0.1,
        markersize=8,
        err_kws={'linewidth': 2},
        ax=ax
    )
    
    ax.set_xlabel("Number of Walkers")
    ax.set_ylabel("Performance (ns/day)")
    #ax.set_xscale("log", base=2)
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
