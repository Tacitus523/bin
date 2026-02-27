#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ase.atoms import Atoms
from ase.io import read

EV_TO_KCAL_PER_MOL = 23.06054783061903
DIST_MIN = 2.0
DIST_MAX = 3.2

sns.set_context("talk", font_scale=1.0)

S_S_INDICES = ((1,6), (1,11))
REF_FORCE_KEYS: Sequence[str] = ("ref_force", "ref_forces", "forces")
PRED_FORCE_KEYS: Sequence[str] = ("pred_forces", "pred_force", "forces_pred")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot 2D surfaces of ref_energy, pred_energy, and their difference "
            "over the two shortest S-S distances."
        )
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to extxyz file containing ref_energy and pred_energy.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="energy_surfaces.png",
        help="Output image path (default: energy_surfaces.png).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="Figure DPI (default: 100).",
    )
    return parser.parse_args()


def get_shortest_ss_pairs_from_reference(atoms: Atoms) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    symbols: List[str] = atoms.get_chemical_symbols()
    sulfur_indices = [index for index, symbol in enumerate(symbols) if symbol == "S"]
    if len(sulfur_indices) < 3:
        raise ValueError("Each frame must contain at least 3 sulfur atoms.")

    positions = atoms.get_positions()
    pair_distances: List[Tuple[float, Tuple[int, int]]] = []
    for first_pos, first_index in enumerate(sulfur_indices):
        for second_index in sulfur_indices[first_pos + 1 :]:
            distance = float(np.linalg.norm(positions[first_index] - positions[second_index]))
            pair_distances.append((distance, (first_index, second_index)))

    pair_distances.sort(key=lambda item: item[0])
    print(f"Shortest S-S pair: indices {pair_distances[0][1]} with distance {pair_distances[0][0]:.3f} Å")
    print(f"Second shortest S-S pair: indices {pair_distances[1][1]} with distance {pair_distances[1][0]:.3f} Å")
    return pair_distances[0][1], pair_distances[1][1]


def get_distance_for_pair(atoms: Atoms, pair: Tuple[int, int]) -> float:
    positions = atoms.get_positions()
    return float(np.linalg.norm(positions[pair[0]] - positions[pair[1]]))


def get_force_arrays(atoms: Atoms) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    ref_key = next((key for key in REF_FORCE_KEYS if key in atoms.arrays), None)
    if ref_key is None:
        return None

    pred_key = next(
        (key for key in PRED_FORCE_KEYS if key in atoms.arrays and key != ref_key),
        None,
    )
    if pred_key is None:
        return None

    ref_forces = np.asarray(atoms.arrays[ref_key], dtype=float)
    pred_forces = np.asarray(atoms.arrays[pred_key], dtype=float)
    if ref_forces.shape != pred_forces.shape or ref_forces.ndim != 2 or ref_forces.shape[1] != 3:
        return None
    return ref_forces, pred_forces


def compute_force_magnitude_rmse(ref_forces: np.ndarray, pred_forces: np.ndarray) -> float:
    ref_magnitudes = np.linalg.norm(ref_forces, axis=1)
    pred_magnitudes = np.linalg.norm(pred_forces, axis=1)
    return float(np.sqrt(np.mean((pred_magnitudes - ref_magnitudes) ** 2)))


def load_surface_data(
    input_file: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frames: List[Atoms] = read(input_file, index=":")

    x_values: List[float] = []
    y_values: List[float] = []
    ref_energies_kcal: List[float] = []
    pred_energies_kcal: List[float] = []
    force_rmse_values: List[float] = []

    reference_pairs: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = S_S_INDICES

    for atoms in frames:
        if "ref_energy" not in atoms.info or "pred_energy" not in atoms.info:
            continue

        if reference_pairs is None:
            reference_pairs = get_shortest_ss_pairs_from_reference(atoms)

        x_values.append(get_distance_for_pair(atoms, reference_pairs[0]))
        y_values.append(get_distance_for_pair(atoms, reference_pairs[1]))
        ref_energies_kcal.append(float(atoms.info["ref_energy"]) * EV_TO_KCAL_PER_MOL)
        pred_energies_kcal.append(float(atoms.info["pred_energy"]) * EV_TO_KCAL_PER_MOL)
        force_arrays = get_force_arrays(atoms)
        if force_arrays is None:
            force_rmse_values.append(np.nan)
        else:
            force_rmse_values.append(compute_force_magnitude_rmse(*force_arrays))

    if not ref_energies_kcal:
        raise ValueError("No frames with both 'ref_energy' and 'pred_energy' found.")

    ref_values = np.array(ref_energies_kcal)
    pred_values = np.array(pred_energies_kcal)
    diff_values = pred_values - ref_values

    ref_values -= np.min(ref_values)
    pred_values -= np.min(pred_values)

    force_rmse_array = np.array(force_rmse_values)
    if not np.any(np.isfinite(force_rmse_array)):
        raise ValueError(
            "No frames with both reference and predicted force arrays found. "
            "Expected keys like ref_force/ref_forces and pred_forces."
        )

    return np.array(x_values), np.array(y_values), ref_values, pred_values, diff_values, force_rmse_array


def get_levels(values: np.ndarray, step: float = 1.0) -> np.ndarray:
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    start = float(np.floor(min_value / step) * step)
    stop = float(np.ceil(max_value / step) * step)
    if start == stop:
        start -= step
        stop += step
    return np.arange(start, stop + step, step)


def plot_one_surface(
    ax: plt.Axes,
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    title: str,
    colorbar_label: str,
    level_step: Optional[float] = 1.0,
) -> None:
    valid_mask = np.isfinite(x_values) & np.isfinite(y_values) & np.isfinite(z_values)
    if np.count_nonzero(valid_mask) < 3:
        raise ValueError(f"Not enough valid points to plot '{title}'.")

    x_plot = x_values[valid_mask]
    y_plot = y_values[valid_mask]
    z_plot = z_values[valid_mask]

    if level_step is None:
        contour_filled = ax.tricontourf(
            x_plot,
            y_plot,
            z_plot,
            cmap="coolwarm",
        )
        ax.tricontour(
            x_plot,
            y_plot,
            z_plot,
            colors="black",
            linewidths=0.6,
            linestyles="solid",
        )
    else:
        line_levels = get_levels(z_plot, step=level_step)
        contour_filled = ax.tricontourf(
            x_plot,
            y_plot,
            z_plot,
            levels=line_levels,
            cmap="coolwarm",
        )
        ax.tricontour(
            x_plot,
            y_plot,
            z_plot,
            levels=line_levels,
            colors="black",
            linewidths=0.6,
            linestyles="solid",
        )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.08)
    colorbar = plt.colorbar(contour_filled, cax=cax)
    colorbar.set_label(colorbar_label)

    ax.set_xlabel(r"S$^\mathrm{1}$-S$^\mathrm{2}$ ($\AA$)")
    ax.set_ylabel(r"S$^\mathrm{2}$-S$^\mathrm{3}$ ($\AA$)")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))


def plot_surface(
    x_values: np.ndarray,
    y_values: np.ndarray,
    ref_values: np.ndarray,
    pred_values: np.ndarray,
    diff_values: np.ndarray,
    force_rmse_values: np.ndarray,
    output_path: Path,
    dpi: int,
) -> None:
    mask = (
        (x_values >= DIST_MIN)
        & (x_values <= DIST_MAX)
        & (y_values >= DIST_MIN)
        & (y_values <= DIST_MAX)
    )
    if not np.any(mask):
        raise ValueError("No data points found in the 2.0-3.2 Å range.")

    x_plot = x_values[mask]
    y_plot = y_values[mask]
    ref_plot = ref_values[mask]
    pred_plot = pred_values[mask]
    diff_plot = diff_values[mask]
    force_rmse_plot = force_rmse_values[mask]

    panels = [
        ("Ref. Energy", ref_plot, "Energy (kcal/mol)", 1.0, "ref"),
        ("Pred. Energy", pred_plot, "Energy (kcal/mol)", 1.0, "pred"),
        ("Pred - Ref", diff_plot, "Energy (kcal/mol)", None, "diff"),
        ("Force Mag. RMSE", force_rmse_plot, r"Force RMSE (eV/$\AA$)", None, "force_rmse"),
    ]

    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(26, 6))
    for axis, (title, values, colorbar_label, level_step, _) in zip(axes, panels):
        plot_one_surface(
            axis,
            x_plot,
            y_plot,
            values,
            title,
            colorbar_label=colorbar_label,
            level_step=level_step,
        )

    for axis in axes:
        axis.set_xlim(DIST_MIN, DIST_MAX)
        axis.set_ylim(DIST_MIN, DIST_MAX)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)

    for title, values, colorbar_label, level_step, suffix in panels:
        fig_single, axis_single = plt.subplots(nrows=1, ncols=1, figsize=(7, 6))
        plot_one_surface(
            axis_single,
            x_plot,
            y_plot,
            values,
            title,
            colorbar_label=colorbar_label,
            level_step=level_step,
        )
        axis_single.set_xlim(DIST_MIN, DIST_MAX)
        axis_single.set_ylim(DIST_MIN, DIST_MAX)
        fig_single.tight_layout()
        individual_path = output_path.with_name(f"{output_path.stem}_{suffix}{output_path.suffix}")
        fig_single.savefig(individual_path, dpi=dpi)
        plt.close(fig_single)


def main() -> None:
    args = parse_args()
    x_values, y_values, ref_values, pred_values, diff_values, force_rmse_values = load_surface_data(args.input_file)
    plot_surface(
        x_values,
        y_values,
        ref_values,
        pred_values,
        diff_values,
        force_rmse_values,
        Path(args.output),
        args.dpi,
    )
    print(f"Saved plot to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
