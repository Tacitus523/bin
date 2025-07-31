#!/usr/bin/env python3
import argparse
from typing import List, Optional, Tuple
import os

from ase import Atoms
from ase.io import read
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

GEOMS_FILE = "geoms_env.extxyz"
ESP_KEY = "esp"
ESP_GRADIENT_KEY = "esp_gradient"

ESP_UNIT = "V"  # Volt
ESP_GRADIENT_UNIT = "V/Å"  # Volt per Angstrom

BINS = 30
ALPHA = 0.7

FIGSIZE = (21,9)
FONTSIZE = 24
LABELSIZE = 20
TITLE = False
DPI = 100

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot histograms and boxplots of ESP and ESP gradients from geometries.")
    ap.add_argument("-g", type=str, dest="geoms_file", required=False, default=GEOMS_FILE, help="File with geometry data", metavar="geometry file")
    ap.add_argument("-s", "--sources", type=str, required=False, default=None, help="File with data sources for coloring", metavar="data source file")
    args = ap.parse_args()
    validate_args(args)
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    return args

def validate_args(args: argparse.Namespace) -> None:
    if not os.path.exists(args.geoms_file):
        raise FileNotFoundError(f"Geometry file '{args.geoms_file}' does not exist.")
    if args.sources and not os.path.exists(args.sources):
        raise FileNotFoundError(f"Data source file '{args.sources}' does not exist.")
    return

def read_geoms(file: str) -> pd.DataFrame:
    geoms: List[Atoms] = read(file, index=":")
    assert len(geoms) > 0, f"No geometries found in file '{file}'."
    assert all(ESP_KEY in geom.arrays for geom in geoms), f"Not all geometries have the '{ESP_KEY}' attribute."
    assert all(ESP_GRADIENT_KEY in geom.arrays for geom in geoms), f"Not all geometries have the '{ESP_GRADIENT_KEY}' attribute."

    elements = []
    esps = []
    esp_gradients = []
    for geom in geoms:
        geom_elements = geom.get_chemical_symbols()
        geom_esp = geom.arrays[ESP_KEY]
        geom_esp_gradient = geom.arrays[ESP_GRADIENT_KEY]
        esp_gradient_manitude = np.linalg.norm(geom_esp_gradient, axis=1)

        elements.append(geom_elements)
        esps.append(geom_esp)
        esp_gradients.append(esp_gradient_manitude)
    
    n_molecules = len(geoms)
    n_atoms = [len(geom) for geom in geoms]
    elements = np.array(elements, dtype=object).flatten()
    esps = np.array(esps).flatten()
    esp_gradients = np.array(esp_gradients).flatten()

    molecule_column = np.concatenate([np.repeat(i, n) for i, n in enumerate(n_atoms)])
    atom_column = np.concatenate([np.arange(n) for n in n_atoms])

    df = pd.DataFrame({
        "Molecule": molecule_column,
        "Atom": atom_column,
        "Element": elements,
        "ESP": esps,
        "ESP Gradient": esp_gradients
    })

    return df


def plot_histogram(data: pd.DataFrame, x: str, unit: str, args: argparse.Namespace ) -> None:
    plt.figure(figsize=FIGSIZE)
    sns.histplot(
        data=data,
        x=x,
        hue="Data Source" if "Data Source" in data.columns else "Element",
        palette="tab10",
        bins=BINS,
        alpha=ALPHA,
        stat="probability",
        common_norm=False,
    )
    if TITLE:
        plt.title(f"{x} Histogram", fontsize=FONTSIZE)

    plt.xlabel(f"{x} [{unit}]")
    plt.ylabel("Probability Density")

    plt.savefig(f"{x}_histogram.png", dpi=DPI)
    plt.close()

def plot_boxplot(data: pd.DataFrame, y: str, unit: str, args: argparse.Namespace) -> None:
    plt.figure(figsize=FIGSIZE)
    hue = "Data Source" if "Data Source" in data.columns else "Element"
    sns.boxplot(
        data=data,
        #x="Element",
        y=y,
        hue=hue,
        palette="tab10"
    )
    if TITLE:
        plt.title(f"{y} Boxplot", fontsize=FONTSIZE)

    #plt.xlabel("Element")
    plt.ylabel(f"{y} [{unit}]")

    plt.savefig(f"{y}_boxplot.png", dpi=DPI)
    plt.close()

def plot_violinplot(data: pd.DataFrame, y: str, unit: str, args: argparse.Namespace) -> None:
    plt.figure(figsize=FIGSIZE)
    hue = "Data Source" if "Data Source" in data.columns else "Element"
    sns.violinplot(
        data=data,
        #x="Element",
        y=y,
        hue=hue,
        palette="tab10"
    )
    if TITLE:
        plt.title(f"{y} Violin Plot", fontsize=FONTSIZE)

    #plt.xlabel("Element")
    plt.ylabel(f"{y} [{unit}]")

    plt.savefig(f"{y}_violin_plot.png", dpi=DPI)
    plt.close()

def main() -> None:
    args = parse_args()
    df = read_geoms(args.geoms_file)
    n_atoms = df.groupby("Molecule")["Atom"].max() + 1
    if args.sources:
        data_source = pd.read_csv(args.sources, sep=",", header=None, names=["Data Source"])
        df["Data Source"] = data_source["Data Source"].repeat(n_atoms).values

    sns.set_context("talk", font_scale=1.3)
    for column, unit in zip(["ESP", "ESP Gradient"], 
                         [ESP_UNIT, ESP_GRADIENT_UNIT]):
        print(f"Creating plots for {column}...")
        plot_histogram(df, column, unit, args)
        plot_boxplot(df, column, unit, args)
        plot_violinplot(df, column, unit, args)

if __name__=="__main__":
    main()
