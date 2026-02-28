#!/usr/bin/env python3
# This script generates a violin plot for ESP (Electrostatic Potential) data extracted from multiple .extxyz files.
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import List, Optional
import warnings

from ase import Atoms
from ase.io import read

ESP_KEY: str = "esp"
ESP_Unit: str = "eV/e" # Unit for ESP, can be adjusted as needed

SAVEPATH: str = "esp_violinplot.png"
FIGSIZE: tuple = (10, 6)
ALPHA: float = 1.0
DPI: int = 300


PALETTE = sns.color_palette("tab10")
PALETTE.pop(3)  # Remove red color

# Silence seaborn UserWarning about palette length
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"The palette list has more values .* than needed .*",
)

sns.set_style("whitegrid")
sns.set_context("talk")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ESP Violin Plot")
    parser.add_argument("files", nargs="+", help="List of .xyz files to plot")
    parser.add_argument("-l", "--labels", nargs="+", type=str, default=None, required=False, help="Labels for each file")
    parser.add_argument("--savepath", default=SAVEPATH, help="Path to save the plot")
    parser.add_argument("--figsize", type=float, nargs=2, default=FIGSIZE, help="Figure size (width, height)")
    parser.add_argument("--alpha", type=float, default=ALPHA, help="Transparency level for the violins")
    args = parser.parse_args()

    if args.labels is not None and len(args.labels) != len(args.files):
        raise ValueError("Number of labels must match number of files.")
    if args.labels is None:
        args.labels = [f"File {i+1}" for i in range(len(args.files))]
    if args.figsize[0] <= 0 or args.figsize[1] <= 0:
        raise ValueError("Figure size must be positive values.")
    return args

def extract_esp_data(file: str) -> np.ndarray:
    """Extract ESP data from an .extxyz file."""
    molecules: List[Atoms] = read(file, format="extxyz", index=":")
    if not ESP_KEY in molecules[0].arrays.keys():
        raise ValueError(f"ESP data not found in {file}. Ensure the file contains 'esp' in its arrays.")

    esps: List[np.ndarray] = []
    for molecule in molecules:
        esp = molecule.arrays[ESP_KEY]
        esps.append(esp)

    return np.concatenate(esps)


def construct_dataframe(data_list: List[np.ndarray], labels: List[str]) -> pd.DataFrame:
    """Construct a DataFrame from the list of ESP data."""
    df_list = []
    for data, label in zip(data_list, labels):
        df = pd.DataFrame()
        df["ESP"] = data
        df["Source"] = label
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)

def plot_violin(df: pd.DataFrame, savepath: str, figsize: tuple, alpha: float) -> None:
    """Plot a violin plot of the ESP data."""
    plt.figure(figsize=figsize)
    sns.violinplot(x="Source", y="ESP", data=df, alpha=alpha, hue="Source", palette=PALETTE)
    plt.xlabel("Source")
    plt.ylabel(f"Electrostatic Potential ({ESP_Unit})")
    plt.ylim(df["ESP"].quantile(0.02), df["ESP"].quantile(0.98))  # Limit y-axis to avoid extreme outliers
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(savepath, dpi=DPI)
    plt.close()

def main() -> None:
    args = parse_args()
    data_list: List[np.ndarray] = []
    
    for file in args.files:
        try:
            data = extract_esp_data(file)
            data_list.append(data)
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue

    if not data_list:
        print("No valid ESP data found. Exiting.")
        return

    df = construct_dataframe(data_list, args.labels)
    plot_violin(df, args.savepath, args.figsize, args.alpha)
    print(f"Violin plot saved to {args.savepath}")

if __name__ == "__main__":
    main()

    
    