#!/lustre/home/ka/ka_ipc/ka_he8978/miniconda3/envs/kgcnn_new/bin/python
"""
Analyze a PLUMED fes.dat file to find local minima and transition states
(saddle points) on the 2D PMF surface. Outputs a CSV with locations and values.
"""
import argparse
import os
from typing import List, Tuple

import heapq
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from scipy.ndimage import minimum_filter


KJ_TO_KCAL = 1 / 4.184
DEFAULT_FES_FILE = "fes.dat"
DEFAULT_OUTPUT = "pmf_critical_points.csv"
# Neighborhood size for local extrema detection (in grid points)
NEIGHBORHOOD_SIZE = 25
# Energy cutoff: ignore points above this (kcal/mol) as likely unsampled
ENERGY_CUTOFF = 20.0
# Number of paths to find per pair of minima
N_PATHS = 3
# Radius (in grid points) to block around a found TS when searching for
# alternative paths
BLOCK_RADIUS = 10
# Minimum barrier height (kcal/mol) above the higher minimum for a TS
# to be considered relevant
TS_MIN_BARRIER = 3.0
# Threshold (in CV units, e.g. degrees) for matching to known conformations
LABEL_RADIUS = 30.0

# Plot settings
PERIODIC_LIMITS = ((-180, 180), (-180, 180))
NON_PERIODIC_LIMITS = ((1.8, 6), (1.8, 6))
FIGSIZE = (7, 6)
DPI = 100
SAVE_PLOT = "PMF_analyzed.png"

# Known conformations for alanine dipeptide (phi, psi in degrees)
# B3LYP/6-311+G(2d,p)//B3LYP/6-31G(d,p)
REFERENCE_MINIMA = {
    "C7eq": (-83.1, 72.6),
    "C5": (-158.4, 164.6),
    "aR": (-80.0, -20.0),
    "aL": (68.4, 26.5),
    "C7ax": (73.6, -57.7),
    "b2": (-125.7, 21.6),
    "a'": (-169.9, -39.2),
    "aD": (59.8, -136.2),
}
REFERENCE_TS = {
    "TS1": (5.6, 81.4),
    "TS2": (-1.4, -8.9),
    "TS3": (2.8, -77.3),
    "TS4": (112.8, -146.7),
    "TS5": (135.9, -26.2),
    "TS6": (79.0, 86.4),
    "TS7": (-149.8, -87.3),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find local minima and transition states on a 2D PMF."
    )
    parser.add_argument(
        "-f", "--fes_file",
        type=str,
        default=DEFAULT_FES_FILE,
        help=f"Path to the fes.dat file (default: {DEFAULT_FES_FILE})",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--energy_cutoff",
        type=float,
        default=ENERGY_CUTOFF,
        help=f"Ignore points above this energy in kcal/mol (default: {ENERGY_CUTOFF})",
    )
    parser.add_argument(
        "--neighborhood",
        type=int,
        default=NEIGHBORHOOD_SIZE,
        help=f"Neighborhood size for extrema detection (default: {NEIGHBORHOOD_SIZE})",
    )
    parser.add_argument(
        "--n_paths",
        type=int,
        default=N_PATHS,
        help=f"Number of paths to find per pair of minima (default: {N_PATHS})",
    )
    parser.add_argument(
        "--block_radius",
        type=int,
        default=BLOCK_RADIUS,
        help=f"Radius around TS to block when searching for alternative paths (default: {BLOCK_RADIUS})",
    )
    parser.add_argument(
        "--ts_min_barrier",
        type=float,
        default=TS_MIN_BARRIER,
        help=f"Minimum barrier height in kcal/mol above the higher minimum (default: {TS_MIN_BARRIER})",
    )
    parser.add_argument(
        "--label_radius",
        type=float,
        default=LABEL_RADIUS,
        help=f"Max CV distance for matching to reference conformations (default: {LABEL_RADIUS})",
    )
    parser.add_argument(
        "--no_reference",
        action="store_true",
        help="Disable reference conformation labeling and TS search",
    )
    parser.add_argument(
        "--e_max",
        type=float,
        default=None,
        help="Max energy for contour levels (kcal/mol). Defaults to energy_cutoff.",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Disable PMF plot generation",
    )
    args = parser.parse_args()
    assert os.path.isfile(args.fes_file), f"Error: {args.fes_file} not found."
    return args


def analyze_fes_header(file_path: str) -> Tuple[List[int], List[bool], List[str]]:
    """Extract nbins, periodicity, and CV names from the fes.dat header."""
    nbins_list: List[int] = []
    periodicity_list: List[bool] = []
    cv_names: List[str] = []
    with open(file_path, "r") as f:
        for line in f:
            if not line.startswith("#!"):
                break
            if "nbins" in line:
                nbins_list.append(int(line.split()[-1]))
                # Extract CV name from e.g. "nbins_ang_1"
                key = line.split()[2]  # e.g. "nbins_ang_1"
                cv_name = key.replace("nbins_", "")
                cv_names.append(cv_name)
            if "periodic" in line:
                periodicity_list.append("true" in line)
    assert len(nbins_list) == 2, f"Expected 2 CVs, found {len(nbins_list)}"
    assert len(periodicity_list) == 2
    return nbins_list, periodicity_list, cv_names


def load_fes_data(
    fes_file_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[bool], List[str]]:
    """Load fes.dat and return 2D grids of CV1, CV2, energy, derivatives."""
    nbins_list, periodicity_list, cv_names = analyze_fes_header(fes_file_path)
    data = np.loadtxt(fes_file_path, usecols=(0, 1, 2, 3, 4), unpack=True)
    x_raw, y_raw, e_raw, dx_raw, dy_raw = data

    xx = np.reshape(x_raw, nbins_list)
    yy = np.reshape(y_raw, nbins_list)
    zz = np.reshape(e_raw * KJ_TO_KCAL, nbins_list)
    # Derivatives: kJ/mol per raw unit -> kcal/mol per display unit
    dzdx = np.reshape(dx_raw, nbins_list)
    dzdy = np.reshape(dy_raw, nbins_list)

    # Convert units for display
    units = []
    for i, periodic in enumerate(periodicity_list):
        if periodic:
            if i == 0:
                xx = xx * 180.0 / np.pi
                # kJ/mol/rad -> kcal/mol/deg
                dzdx = dzdx * KJ_TO_KCAL * np.pi / 180.0
            else:
                yy = yy * 180.0 / np.pi
                dzdy = dzdy * KJ_TO_KCAL * np.pi / 180.0
            units.append("deg")
        else:
            if i == 0:
                xx = xx * 10.0  # nm -> Angstrom
                # kJ/mol/nm -> kcal/mol/A
                dzdx = dzdx * KJ_TO_KCAL / 10.0
            else:
                yy = yy * 10.0
                dzdy = dzdy * KJ_TO_KCAL / 10.0
            units.append("A")

    return xx, yy, zz, dzdx, dzdy, periodicity_list, cv_names


def find_local_minima(
    zz: np.ndarray, neighborhood: int, energy_cutoff: float, periodic: List[bool]
) -> np.ndarray:
    """Find local minima on the 2D energy surface. Returns boolean mask."""
    zz_work = _pad_periodic(zz, neighborhood, periodic)
    local_min = minimum_filter(zz_work, size=neighborhood, mode="nearest")
    # Crop back to original shape
    local_min = _crop_periodic(local_min, zz.shape, neighborhood, periodic)
    mask = (zz == local_min) & (zz < energy_cutoff)
    return mask


def find_transition_states(
    xx: np.ndarray,
    yy: np.ndarray,
    zz: np.ndarray,
    dzdx: np.ndarray,
    dzdy: np.ndarray,
    minima_mask: np.ndarray,
    periodic: List[bool],
    cv_names: List[str],
    n_paths: int = 1,
    block_radius: int = BLOCK_RADIUS,
    energy_cutoff: float = ENERGY_CUTOFF,
    ts_min_barrier: float = TS_MIN_BARRIER,
) -> pd.DataFrame:
    """
    Find transition states via minimax pathfinding between all pairs of minima.
    For each pair, find up to n_paths paths by iteratively blocking the TS
    region of previously found paths and re-running the search.
    """
    minima_coords = list(zip(*np.where(minima_mask)))
    n_min = len(minima_coords)
    if n_min < 2:
        return pd.DataFrame()

    # Collect all TS rows; deduplicate by grid location
    ts_dict: dict = {}  # (i_row, j_col) -> row dict

    for (idx_a, min_a), (idx_b, min_b) in combinations(enumerate(minima_coords), 2):
        label_a = f"min{idx_a}"
        label_b = f"min{idx_b}"
        # Energy of the higher minimum in this pair
        e_min_a = float(zz[min_a])
        e_min_b = float(zz[min_b])
        e_higher_min = max(e_min_a, e_min_b)
        # Work on a copy so blocking doesn't affect other pairs
        zz_work = zz.copy()

        for path_idx in range(n_paths):
            ts_ij, ts_energy, path = _minimax_path(zz_work, min_a, min_b, periodic)
            if ts_ij is None:
                break
            # Skip if TS energy exceeds cutoff
            if ts_energy > energy_cutoff:
                break
            # Skip if barrier height above the higher minimum is too small
            if ts_energy - e_higher_min < ts_min_barrier:
                # Still block and continue to find alternative routes
                _block_region(zz_work, ts_ij, block_radius, periodic)
                continue

            key = ts_ij
            row = {
                "type": "transition_state",
                cv_names[0]: round(float(xx[ts_ij]), 2),
                cv_names[1]: round(float(yy[ts_ij]), 2),
                "energy_kcal_mol": round(float(ts_energy), 2),
                f"der_{cv_names[0]}": round(float(dzdx[ts_ij]), 4),
                f"der_{cv_names[1]}": round(float(dzdy[ts_ij]), 4),
                "barrier_kcal_mol": round(float(ts_energy - e_higher_min), 2),
                "connects": f"{label_a}<->{label_b}",
            }
            if key not in ts_dict or ts_energy < ts_dict[key]["energy_kcal_mol"]:
                ts_dict[key] = row

            # Block the region around this TS so the next search finds
            # an alternative route
            _block_region(zz_work, ts_ij, block_radius, periodic)

    ts_df = pd.DataFrame(list(ts_dict.values()))
    if not ts_df.empty:
        ts_df.insert(1, "label", [f"ts{i}" for i in range(len(ts_df))])
    return ts_df


def _periodic_cv_distance(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    periodic: List[bool],
    period: Tuple[float, float] = (360.0, 360.0),
) -> float:
    """Compute distance between two CV points, handling periodicity."""
    d = [0.0, 0.0]
    for i in range(2):
        diff = abs(p1[i] - p2[i])
        if periodic[i]:
            diff = min(diff, period[i] - diff)
        d[i] = diff
    return np.sqrt(d[0] ** 2 + d[1] ** 2)


def _find_nearest_grid_point(
    xx: np.ndarray,
    yy: np.ndarray,
    target: Tuple[float, float],
    periodic: List[bool],
) -> Tuple[int, int]:
    """Find grid indices nearest to target point, handling periodicity."""
    dx = xx - target[0]
    dy = yy - target[1]
    if periodic[0]:
        dx = np.minimum(np.abs(dx), 360.0 - np.abs(dx))
    else:
        dx = np.abs(dx)
    if periodic[1]:
        dy = np.minimum(np.abs(dy), 360.0 - np.abs(dy))
    else:
        dy = np.abs(dy)
    dist = np.sqrt(dx ** 2 + dy ** 2)
    idx = np.unravel_index(np.argmin(dist), dist.shape)
    return idx


def _block_region(
    zz: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    periodic: List[bool],
) -> None:
    """Set energy to infinity in a circular region around center (in-place)."""
    nr, nc = zz.shape
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr * dr + dc * dc > radius * radius:
                continue
            r = center[0] + dr
            c = center[1] + dc
            if periodic[0]:
                r = r % nr
            elif r < 0 or r >= nr:
                continue
            if periodic[1]:
                c = c % nc
            elif c < 0 or c >= nc:
                continue
            zz[r, c] = np.inf


def _minimax_path(
    zz: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    periodic: List[bool],
) -> Tuple:
    """
    Modified Dijkstra: find path from start to end that minimizes the
    maximum energy along the path (bottleneck shortest path).
    Returns (ts_grid_index, ts_energy, path) or (None, None, None).
    """
    nr, nc = zz.shape
    # Priority queue: (max_energy_along_path, row, col)
    heap: list = [(float(zz[start]), start[0], start[1])]
    # Best known bottleneck cost to reach each cell
    best = np.full((nr, nc), np.inf)
    best[start] = zz[start]
    # Parent tracking for path reconstruction
    parent = np.full((nr, nc, 2), -1, dtype=int)

    # 8-connected neighbors
    offsets = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1),          (0, 1),
               (1, -1),  (1, 0), (1, 1)]

    while heap:
        cost, r, c = heapq.heappop(heap)
        if (r, c) == end:
            break
        if cost > best[r, c]:
            continue
        for dr, dc in offsets:
            nr2 = r + dr
            nc2 = c + dc
            # Handle periodicity
            if periodic[0]:
                nr2 = nr2 % nr
            elif nr2 < 0 or nr2 >= nr:
                continue
            if periodic[1]:
                nc2 = nc2 % nc
            elif nc2 < 0 or nc2 >= nc:
                continue

            new_cost = max(cost, float(zz[nr2, nc2]))
            if new_cost < best[nr2, nc2]:
                best[nr2, nc2] = new_cost
                parent[nr2, nc2] = [r, c]
                heapq.heappush(heap, (new_cost, nr2, nc2))

    if best[end] == np.inf:
        return None, None, None

    # Reconstruct path and find highest point (transition state)
    path = []
    cur = list(end)
    while cur[0] != -1:
        path.append((cur[0], cur[1]))
        if (cur[0], cur[1]) == start:
            break
        pr, pc = parent[cur[0], cur[1]]
        cur = [pr, pc]
    path.reverse()

    # Find the grid point with highest energy along the path
    energies = [zz[p] for p in path]
    ts_idx = int(np.argmax(energies))
    ts_ij = path[ts_idx]
    ts_energy = energies[ts_idx]
    return ts_ij, ts_energy, path


def _pad_periodic(
    zz: np.ndarray, pad: int, periodic: List[bool]
) -> np.ndarray:
    """Pad array for periodic boundary handling."""
    if periodic[0]:
        zz = np.concatenate([zz[-pad:, :], zz, zz[:pad, :]], axis=0)
    if periodic[1]:
        zz = np.concatenate([zz[:, -pad:], zz, zz[:, :pad]], axis=1)
    return zz


def _crop_periodic(
    arr: np.ndarray, orig_shape: Tuple[int, int], pad: int, periodic: List[bool]
) -> np.ndarray:
    """Crop padded array back to original shape."""
    if periodic[0]:
        arr = arr[pad : pad + orig_shape[0], :]
    if periodic[1]:
        arr = arr[:, pad : pad + orig_shape[1]]
    return arr


def assign_literature_labels(
    df: pd.DataFrame,
    cv_names: List[str],
    periodic: List[bool],
    label_radius: float = LABEL_RADIUS,
) -> pd.DataFrame:
    """Match found critical points to known reference conformations."""
    for idx, row in df.iterrows():
        point = (row[cv_names[0]], row[cv_names[1]])
        if row["type"] == "minimum":
            ref_set = REFERENCE_MINIMA
        else:
            ref_set = REFERENCE_TS

        best_name = None
        best_dist = float("inf")
        for name, ref_point in ref_set.items():
            dist = _periodic_cv_distance(point, ref_point, periodic)
            if dist < best_dist:
                best_dist = dist
                best_name = name

        if best_dist <= label_radius:
            df.at[idx, "label"] = best_name
    return df


def _is_saddle_point(
    zz: np.ndarray,
    idx: Tuple[int, int],
    periodic: List[bool],
) -> bool:
    """Check if a grid point is a saddle point via det(Hessian) < 0."""
    nr, nc = zz.shape
    r, c = idx

    def _get(dr: int, dc: int) -> float:
        ri = (r + dr) % nr if periodic[0] else r + dr
        ci = (c + dc) % nc if periodic[1] else c + dc
        if not periodic[0] and (ri < 0 or ri >= nr):
            return np.nan
        if not periodic[1] and (ci < 0 or ci >= nc):
            return np.nan
        return float(zz[ri, ci])

    f00 = float(zz[r, c])
    fp0 = _get(1, 0)
    fm0 = _get(-1, 0)
    f0p = _get(0, 1)
    f0m = _get(0, -1)
    fpp = _get(1, 1)
    fmm = _get(-1, -1)
    fpm = _get(1, -1)
    fmp = _get(-1, 1)

    if any(np.isnan(v) for v in [fp0, fm0, f0p, f0m, fpp, fmm, fpm, fmp]):
        return False

    hxx = fp0 - 2 * f00 + fm0
    hyy = f0p - 2 * f00 + f0m
    hxy = (fpp - fpm - fmp + fmm) / 4.0
    det_h = hxx * hyy - hxy * hxy
    return det_h < 0


def find_reference_ts(
    xx: np.ndarray,
    yy: np.ndarray,
    zz: np.ndarray,
    dzdx: np.ndarray,
    dzdy: np.ndarray,
    periodic: List[bool],
    cv_names: List[str],
    energy_cutoff: float,
    found_ts_labels: set,
    search_radius: int = BLOCK_RADIUS,
) -> pd.DataFrame:
    """Search for saddle points near known TS locations not already found."""
    nr, nc = zz.shape
    rows = []
    for name, (cv1_ref, cv2_ref) in REFERENCE_TS.items():
        if name in found_ts_labels:
            continue
        center = _find_nearest_grid_point(xx, yy, (cv1_ref, cv2_ref), periodic)

        # Search neighborhood for saddle points (det(H) < 0)
        saddle_candidates = []
        for dr in range(-search_radius, search_radius + 1):
            for dc in range(-search_radius, search_radius + 1):
                if dr * dr + dc * dc > search_radius * search_radius:
                    continue
                r = center[0] + dr
                c = center[1] + dc
                if periodic[0]:
                    r = r % nr
                elif r < 0 or r >= nr:
                    continue
                if periodic[1]:
                    c = c % nc
                elif c < 0 or c >= nc:
                    continue
                if zz[r, c] > energy_cutoff:
                    continue
                if _is_saddle_point(zz, (r, c), periodic):
                    cv_point = (float(xx[r, c]), float(yy[r, c]))
                    dist = _periodic_cv_distance(
                        cv_point, (cv1_ref, cv2_ref), periodic
                    )
                    saddle_candidates.append(((r, c), float(zz[r, c]), dist))

        if saddle_candidates:
            # Pick the saddle point closest to the reference location
            saddle_candidates.sort(key=lambda x: x[2])
            best_idx, best_energy, _ = saddle_candidates[0]
        else:
            raise ValueError(f"The transition state {name} was not found")
            # Fall back to the energy at the reference grid point
            best_idx = center
            best_energy = float(zz[center])
            if best_energy > energy_cutoff:
                continue

        rows.append(
            {
                "type": "reference_ts",
                "label": name,
                cv_names[0]: round(float(xx[best_idx]), 2),
                cv_names[1]: round(float(yy[best_idx]), 2),
                "energy_kcal_mol": round(best_energy, 2),
                f"der_{cv_names[0]}": round(float(dzdx[best_idx]), 4),
                f"der_{cv_names[1]}": round(float(dzdy[best_idx]), 4),
                "barrier_kcal_mol": np.nan,
                "connects": "",
            }
        )
    return pd.DataFrame(rows)


def plot_pmf(
    xx: np.ndarray,
    yy: np.ndarray,
    zz: np.ndarray,
    periodic: List[bool],
    cv_names: List[str],
    df: pd.DataFrame,
    e_max: float,
    save_file: str = SAVE_PLOT,
) -> None:
    """Plot the PMF contour with minima and transition states marked."""
    # Tile periodic data for seamless plotting
    xx_plot, yy_plot, zz_plot = xx.copy(), yy.copy(), zz.copy()
    if periodic[0]:
        xx_plot = np.concatenate(
            [xx_plot - 360, xx_plot, xx_plot + 360], axis=1
        )
        yy_plot = np.concatenate([yy_plot, yy_plot, yy_plot], axis=1)
        zz_plot = np.concatenate([zz_plot, zz_plot, zz_plot], axis=1)
    if periodic[1]:
        xx_plot = np.concatenate([xx_plot, xx_plot, xx_plot], axis=0)
        yy_plot = np.concatenate(
            [yy_plot - 360, yy_plot, yy_plot + 360], axis=0
        )
        zz_plot = np.concatenate([zz_plot, zz_plot, zz_plot], axis=0)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    label_fs = 18
    ticks_fs = 16
    labelpad = -0.8

    # Axis limits and labels
    if periodic[0]:
        ax.set_xlim(*PERIODIC_LIMITS[0])
        ax.set_xlabel(r"$\phi$ (°)", fontsize=label_fs, labelpad=labelpad)
        ax.xaxis.set_major_locator(plt.MultipleLocator(60))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(10))
    else:
        ax.set_xlim(*NON_PERIODIC_LIMITS[0])
        ax.set_xlabel(
            r"S$^\mathrm{1}$-S$^\mathrm{2}$ ($\AA$)",
            fontsize=label_fs,
            labelpad=labelpad,
        )
    if periodic[1]:
        ax.set_ylim(*PERIODIC_LIMITS[1])
        ax.set_ylabel(r"$\psi$ (°)", fontsize=label_fs, labelpad=labelpad)
        ax.yaxis.set_major_locator(plt.MultipleLocator(60))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(10))
    else:
        ax.set_ylim(*NON_PERIODIC_LIMITS[1])
        ax.set_ylabel(
            r"S$^\mathrm{2}$-S$^\mathrm{3}$ ($\AA$)",
            fontsize=label_fs,
            labelpad=labelpad,
        )

    ax.tick_params(axis="x", labelsize=ticks_fs)
    ax.tick_params(axis="y", labelsize=ticks_fs)
    ax.set_aspect("equal")

    # Contour levels
    lines = np.arange(0, e_max + 0.1, 1.0)
    linesf = np.arange(0, e_max + 0.1, 0.1)
    cbticks = np.arange(0, e_max + 0.1, 5)

    cf = ax.contourf(xx_plot, yy_plot, zz_plot, cmap="viridis", levels=linesf)
    ax.contour(
        xx_plot, yy_plot, zz_plot, levels=lines, colors="black", linewidths=0.3
    )
    cb = plt.colorbar(cf, ticks=cbticks, ax=ax, pad=0.02, fraction=0.053, aspect=20)
    cb.set_label(
        r"$\Delta$G (kcal/mol)", fontsize=label_fs, labelpad=5, rotation=90
    )
    cb.ax.tick_params(labelsize=ticks_fs)

    # Mark critical points
    if not df.empty:
        marker_cfg = {
            "minimum": {"marker": "*", "color": "white", "edgecolor": "black", "s": 200, "zorder": 5},
            "transition_state": {"marker": "^", "color": "red", "edgecolor": "black", "s": 150, "zorder": 5},
            "reference_ts": {"marker": "^", "color": "orange", "edgecolor": "black", "s": 150, "zorder": 5},
        }
        for _, row in df.iterrows():
            cfg = marker_cfg.get(row["type"], marker_cfg["transition_state"])
            cv1 = row[cv_names[0]]
            cv2 = row[cv_names[1]]
            ax.scatter(cv1, cv2, **cfg)
            label = row.get("label", "")
            if label:
                ax.annotate(
                    label,
                    (cv1, cv2),
                    textcoords="offset points",
                    xytext=(6, 6),
                    fontsize=9,
                    fontweight="bold",
                    color="white",
                    path_effects=[
                        patheffects.withStroke(
                            linewidth=2, foreground="black"
                        )
                    ],
                )

    fig.tight_layout()
    plt.savefig(save_file, dpi=DPI, bbox_inches="tight")
    print(f"Plot saved to {save_file}")
    plt.close(fig)


def build_results_table(
    xx: np.ndarray,
    yy: np.ndarray,
    zz: np.ndarray,
    dzdx: np.ndarray,
    dzdy: np.ndarray,
    minima_mask: np.ndarray,
    ts_df: pd.DataFrame,
    cv_names: List[str],
) -> pd.DataFrame:
    """Build a DataFrame of critical points sorted by energy."""
    rows = []
    for i, idx in enumerate(zip(*np.where(minima_mask))):
        rows.append(
            {
                "type": "minimum",
                "label": f"min{i}",
                cv_names[0]: round(float(xx[idx]), 2),
                cv_names[1]: round(float(yy[idx]), 2),
                "energy_kcal_mol": round(float(zz[idx]), 2),
                f"der_{cv_names[0]}": round(float(dzdx[idx]), 4),
                f"der_{cv_names[1]}": round(float(dzdy[idx]), 4),
                "connects": "",
            }
        )
    min_df = pd.DataFrame(rows)
    df = pd.concat([min_df, ts_df], ignore_index=True)
    if not df.empty:
        df = df.sort_values("energy_kcal_mol").reset_index(drop=True)
    return df


def main() -> None:
    args = parse_args()
    xx, yy, zz, dzdx, dzdy, periodicity, cv_names = load_fes_data(args.fes_file)

    # Shift so global minimum = 0
    zz -= zz.min()

    minima_mask = find_local_minima(zz, args.neighborhood, args.energy_cutoff, periodicity)
    # ts_df = find_transition_states(
    #     xx, yy, zz, minima_mask, periodicity, cv_names,
    #     n_paths=args.n_paths, block_radius=args.block_radius,
    #     energy_cutoff=args.energy_cutoff, ts_min_barrier=args.ts_min_barrier,
    # )
    ts_df = pd.DataFrame()

    df = build_results_table(xx, yy, zz, dzdx, dzdy, minima_mask, ts_df, cv_names)

    if not args.no_reference:
        # Label found points with literature names
        df = assign_literature_labels(df, cv_names, periodicity, args.label_radius)
        # Search for reference TS not found by pathfinding
        found_ts_labels = set(
            df.loc[df["type"] == "transition_state", "label"].dropna()
        )
        ref_ts_df = find_reference_ts(
            xx, yy, zz, dzdx, dzdy, periodicity, cv_names,
            args.energy_cutoff, found_ts_labels,
        )
        if not ref_ts_df.empty:
            df = pd.concat([df, ref_ts_df], ignore_index=True)
            df = df.sort_values("energy_kcal_mol").reset_index(drop=True)

    df["type"] = pd.Categorical(values=df["type"], ordered=True, categories=["minimum", "transition_state", "reference_ts"])
    df = df.sort_values(["type", "label"]).reset_index(drop=True)

    n_min = (df["type"] == "minimum").sum()
    n_ts = (df["type"] == "transition_state").sum()
    n_ref = (df["type"] == "reference_ts").sum()
    df.to_csv(args.output, index=False)
    print(f"Found {n_min} minima, {n_ts} transition states"
          f"{f', {n_ref} reference TS' if n_ref else ''}.")
    print(f"Results saved to {args.output}")
    print()
    print(df.to_string(index=False))

    if not args.no_plot:
        e_max = args.e_max if args.e_max is not None else args.energy_cutoff
        plot_pmf(xx, yy, zz, periodicity, cv_names, df, e_max)


if __name__ == "__main__":
    main()
