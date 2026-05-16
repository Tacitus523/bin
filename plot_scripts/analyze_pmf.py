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
N_PATHS = 1
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
NON_PERIODIC_LIMITS = ((1.8, 4), (1.8, 4))
FIGSIZE = (7, 6)
DPI = 100
SAVE_PLOT = "PMF_analyzed.png"

# Known conformations for alanine dipeptide in vacuum (phi, psi in degrees)
# Each system maps to {"minima": {...}, "ts": {...}}
REFERENCE_SYSTEMS = {
    # B3LYP/6-311+G(2d,p)//B3LYP/6-31G(d,p) dipetid vacuum
    # https://pubs.acs.org/doi/10.1021/ct100395n
    "dipeptid_vacuum": {
        "minima": {
            r"C7$_{eq}$": (-83.1, 72.6),
            r"C$_5$": (-158.4, 164.6),
            r"$\alpha_R$": (-80.0, -20.0),
            r"$\alpha_L$": (68.4, 26.5),
            r"C7$_{ax}$": (73.6, -57.7),
            r"$\beta_2$": (-125.7, 21.6),
            r"$\alpha'$": (-169.9, -39.2),
            r"$\alpha_D$": (59.8, -136.2),
        },
        "ts": {
            "TS1": (5.6, 81.4),
            "TS2": (-1.4, -8.9),
            "TS3": (2.8, -77.3),
            "TS4": (112.8, -146.7),
            "TS5": (135.9, -26.2),
            "TS6": (79.0, 86.4),
            "TS7": (-149.8, -87.3),
        },
    },
    # B3LYP/6-311+G(2d,p)//B3LYP/6-31G(d,p) — dipeptid water
    # https://pubs.acs.org/doi/10.1021/ct100395n
    "dipeptid_water": {
        "minima": {
            r"C7$_{eq}$": (-85.4, 73.4),
            r"C$_5$": (-151.6, 147.6),
            r"$\alpha_R$": (-78.1, -27.2),
            r"$\beta$": (-75.1, 143.3),
            r"$\alpha_L$": (61.3, 40.9),
            r"C7$_{ax}$": (73.4, -53.0),
            r"$\beta_2$": (-138.5, 27.3),
            r"$\alpha_D$": (60.1, -147.7),
            r"$\alpha_D'$": (72.8, 164.8),
        },
        "ts": {
            "TS0": (-129.8, 62.6),
            "TS1": (0.3, 91.6),
            "TS2": (-11.0, -11.7),
            "TS3": (7.3, -92.1),
            "TS5": (132.2, -28.1),
            "TS6": (81.3, 104.2),
            "TS7": (-114.0, -115.6),
            "TS8": (127.9, 133.7),
        },
    },
    "thiol_disulfide_vacuum": {
            "minima": {
                "Reactant": (2.1, 3.1),
                "Product": (3.1, 2.1),
            },
            "ts": {
                "TS": (2.5, 2.5),
            },
    },
    "thiol_disulfide_water": {
            "minima": {
                "Reactant": (2.2, 3.2),
                "Product": (3.2, 2.2),
            },
            "ts": {
                "TS": (2.5, 2.5),
            },
    },
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
        "--system",
        type=str,
        default="dipeptid_vacuum",
        choices=list(REFERENCE_SYSTEMS.keys()),
        help=f"Reference system for labeling (choices: {list(REFERENCE_SYSTEMS.keys())}, default: dipeptid_vacuum)",
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
    parser.add_argument(
        "--one_minimum_per_label",
        action="store_true",
        help="Keep only the lowest-energy minimum for each assigned reference label.",
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
            ts_ij, ts_energy, path = _compute_mep(zz_work, min_a, min_b, periodic)
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

    def _is_forbidden_nonperiodic_boundary(cell: Tuple[int, int]) -> bool:
        row, col = cell
        on_row_boundary = (row == 0 or row == nr - 1)
        on_col_boundary = (col == 0 or col == nc - 1)
        forbidden_row = (not periodic[0]) and on_row_boundary
        forbidden_col = (not periodic[1]) and on_col_boundary
        return forbidden_row or forbidden_col

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

            candidate = (nr2, nc2)
            if candidate != start and candidate != end and _is_forbidden_nonperiodic_boundary(candidate):
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


def _downhill_path_to(
    zz: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    periodic: List[bool],
) -> List[Tuple[int, int]]:
    """
    Find the path from start to end that minimizes total cumulative uphill cost
    (Dijkstra where cost = sum of max(0, dE) for each step).
    Strongly prefers downhill moves but always reaches the target.
    """
    nr, nc = zz.shape
    offsets = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1),           (0, 1),
               (1, -1),  (1, 0),  (1, 1)]
    heap: list = [(0.0, start[0], start[1])]
    best = np.full((nr, nc), np.inf)
    best[start] = 0.0
    parent = np.full((nr, nc, 2), -1, dtype=int)

    while heap:
        cost, r, c = heapq.heappop(heap)
        if (r, c) == end:
            break
        if cost > best[r, c]:
            continue
        for dr, dc in offsets:
            r2 = r + dr
            c2 = c + dc
            if periodic[0]:
                r2 = r2 % nr
            elif r2 < 0 or r2 >= nr:
                continue
            if periodic[1]:
                c2 = c2 % nc
            elif c2 < 0 or c2 >= nc:
                continue
            step_cost = max(0.0, float(zz[r2, c2]) - float(zz[r, c]))
            new_cost = cost + step_cost
            if new_cost < best[r2, c2]:
                best[r2, c2] = new_cost
                parent[r2, c2] = [r, c]
                heapq.heappush(heap, (new_cost, r2, c2))

    # Reconstruct path
    path = []
    cur = list(end)
    while cur[0] != -1:
        path.append((cur[0], cur[1]))
        if (cur[0], cur[1]) == start:
            break
        pr, pc = parent[cur[0], cur[1]]
        cur = [pr, pc]
    path.reverse()
    return path


def _compute_mep(
    zz: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    periodic: List[bool],
) -> Tuple:
    """
    Compute the minimum energy path (MEP) from start to end.
    Uses minimax to locate the TS (bottleneck), then least-uphill Dijkstra
    from the TS to each endpoint to trace the valley floors correctly.
    Returns (ts_grid_index, ts_energy, path) or (None, None, None).
    """
    ts_ij, ts_energy, mm_path = _minimax_path(zz, start, end, periodic)
    if mm_path is None:
        return None, None, None

    ts_idx = next(i for i, p in enumerate(mm_path) if p == ts_ij)

    # Least-uphill path from start to cell just before TS (reversed)
    if ts_idx > 0:
        pre_path = _downhill_path_to(zz, start, mm_path[ts_idx - 1], periodic)
    else:
        pre_path = []

    # Least-uphill path from cell just after TS to end
    if ts_idx < len(mm_path) - 1:
        post_path = _downhill_path_to(zz, mm_path[ts_idx + 1], end, periodic)
    else:
        post_path = []

    full_path = pre_path + [ts_ij] + post_path
    return ts_ij, ts_energy, full_path


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
    ref_minima: dict,
    ref_ts: dict,
    label_radius: float = LABEL_RADIUS,
) -> pd.DataFrame:
    """Match found critical points to known reference conformations."""
    for idx, row in df.iterrows():
        point = (row[cv_names[0]], row[cv_names[1]])
        if row["type"] == "minimum":
            ref_set = ref_minima
        else:
            ref_set = ref_ts

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

def _deduplicate_minima_by_label(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    minima = df[df["type"] == "minimum"].copy()
    others = df[df["type"] != "minimum"].copy()

    labeled = minima[minima["label"].notna() & (minima["label"] != "")]
    unlabeled = minima[~(minima["label"].notna() & (minima["label"] != ""))]

    labeled = labeled.sort_values("energy_kcal_mol").drop_duplicates(
        subset=["label"], keep="first"
    )

    out = pd.concat([labeled, unlabeled, others], ignore_index=True)
    return out.sort_values("energy_kcal_mol").reset_index(drop=True)

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
    ref_ts: dict,
    search_radius: int = BLOCK_RADIUS,
) -> pd.DataFrame:
    """Search for saddle points near known TS locations not already found."""
    nr, nc = zz.shape
    rows = []
    for name, (cv1_ref, cv2_ref) in ref_ts.items():
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
    mep_path: List[Tuple[int, int]] = None,
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

    # Draw MEP path if provided
    if mep_path is not None:
        path_cv1 = np.array([float(xx[p]) for p in mep_path])
        path_cv2 = np.array([float(yy[p]) for p in mep_path])
        ax.plot(path_cv1, path_cv2, color="white", linewidth=1.5,
                linestyle="--", zorder=4, alpha=0.8)

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
                    fontsize=14,
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


def plot_reaction_coordinate(
    xx: np.ndarray,
    yy: np.ndarray,
    zz: np.ndarray,
    df: pd.DataFrame,
    cv_names: List[str],
    periodic: List[bool],
    save_file: str = "reaction_coordinate.png",
    mep_path: List[Tuple[int, int]] = None,
) -> None:
    """Project 2D PMF onto the MEP between 2 minima, plot energy vs reaction coordinate."""
    minima = df[df["type"] == "minimum"].reset_index(drop=True)
    ts_rows = df[df["type"].isin(["transition_state", "reference_ts"])].reset_index(drop=True)

    idx_a = _find_nearest_grid_point(
        xx, yy,
        (float(minima.iloc[0][cv_names[0]]), float(minima.iloc[0][cv_names[1]])),
        periodic,
    )
    idx_b = _find_nearest_grid_point(
        xx, yy,
        (float(minima.iloc[1][cv_names[0]]), float(minima.iloc[1][cv_names[1]])),
        periodic,
    )

    _, _, path = _compute_mep(zz, idx_a, idx_b, periodic)
    if path is None:
        print("Could not find MEP for reaction coordinate projection.")
        return

    mep_path = mep_path if mep_path is not None else path

    energies = np.array([float(zz[p]) for p in mep_path])
    coords_x = np.array([float(xx[p]) for p in mep_path])
    coords_y = np.array([float(yy[p]) for p in mep_path])

    dx = np.diff(coords_x)
    dy = np.diff(coords_y)
    if periodic[0]:
        dx = np.where(np.abs(dx) > 180, dx - np.sign(dx) * 360, dx)
    if periodic[1]:
        dy = np.where(np.abs(dy) > 180, dy - np.sign(dy) * 360, dy)
    arc = np.concatenate([[0.0], np.cumsum(np.sqrt(dx**2 + dy**2))])

    ts_path_idx = int(np.argmax(energies))
    e_a, e_b, e_ts = energies[0], energies[-1], energies[ts_path_idx]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=DPI)
    ax.plot(arc, energies, color="steelblue", linewidth=2, zorder=2)

    # Horizontal dashed reference lines at minima and TS
    for e_level in (e_a, e_b, e_ts):
        ax.axhline(e_level, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

    # Minima markers
    for i, (path_edge_idx, min_row) in enumerate([(0, minima.iloc[0]), (-1, minima.iloc[1])]):
        lbl = str(min_row.get("label", f"min{i}"))
        pos_x, pos_y = arc[path_edge_idx], energies[path_edge_idx]
        ax.scatter(pos_x, pos_y, marker="*", s=300, color="white", edgecolor="black", zorder=5)
        ax.annotate(lbl, (pos_x, pos_y), textcoords="offset points",
                    xytext=(6, 8), fontsize=12, fontweight="bold")

    # TS marker
    lbl_ts = str(ts_rows.iloc[0]["label"]) if len(ts_rows) > 0 else "TS"
    ax.scatter(arc[ts_path_idx], e_ts, marker="^", s=150, color="red", edgecolor="black", zorder=5)
    ax.annotate(lbl_ts, (arc[ts_path_idx], e_ts),
                textcoords="offset points", xytext=(6, 8), fontsize=12, fontweight="bold")

    # Forward barrier annotation (A -> TS)
    x_fwd = (arc[0] + arc[ts_path_idx]) / 2
    ax.annotate("", xy=(x_fwd, e_ts), xytext=(x_fwd, e_a),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.2))
    ax.text(x_fwd + 0.01 * arc[-1], (e_a + e_ts) / 2,
            f"{e_ts - e_a:.1f} kcal/mol", fontsize=10, va="center")

    # Backward barrier annotation (B -> TS)
    x_bwd = (arc[ts_path_idx] + arc[-1]) / 2
    ax.annotate("", xy=(x_bwd, e_ts), xytext=(x_bwd, e_b),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.2))
    ax.text(x_bwd + 0.01 * arc[-1], (e_b + e_ts) / 2,
            f"{e_ts - e_b:.1f} kcal/mol", fontsize=10, va="center")

    ax.set_xlabel(f"Reaction Coordinate", fontsize=14)
    ax.set_ylabel(r"$\Delta$G (kcal/mol)", fontsize=14)
    ax.set_ylim(bottom=0, top=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(save_file, dpi=DPI, bbox_inches="tight")
    print(f"Reaction coordinate plot saved to {save_file}")
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
    ts_df = find_transition_states(
        xx, yy, zz, dzdx, dzdy,
        minima_mask, periodicity, cv_names,
        n_paths=args.n_paths, block_radius=args.block_radius,
        energy_cutoff=args.energy_cutoff, ts_min_barrier=args.ts_min_barrier,
    )
    ts_df = pd.DataFrame() # Reset TS results for pathfinding free analysis

    df = build_results_table(xx, yy, zz, dzdx, dzdy, minima_mask, ts_df, cv_names)

    if not args.no_reference:
        ref_system = REFERENCE_SYSTEMS[args.system]
        ref_minima = ref_system["minima"]
        ref_ts = ref_system["ts"]
        df = assign_literature_labels(
            df, cv_names, periodicity, ref_minima, ref_ts, args.label_radius
        )

        if args.one_minimum_per_label:
            df = _deduplicate_minima_by_label(df)

        found_ts_labels = set(
            df.loc[df["type"] == "transition_state", "label"].dropna()
        )
        ref_ts_df = find_reference_ts(
            xx, yy, zz, dzdx, dzdy, periodicity, cv_names,
            args.energy_cutoff, found_ts_labels, ref_ts,
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
    print(df.to_string(index=False))

    # # drop transition states for plotting
    # df = df[df["type"] == "minimum"].reset_index(drop=True)

    if not args.no_plot:
        e_max = args.e_max if args.e_max is not None else args.energy_cutoff
        mep_path = None
        if not args.no_reference:
            ref_system = REFERENCE_SYSTEMS[args.system]
            if len(ref_system["minima"]) == 2 and len(ref_system["ts"]) == 1:
                minima_df = df[df["type"] == "minimum"].reset_index(drop=True)

                idx_a = _find_nearest_grid_point(
                    xx, yy,
                    list(ref_system["minima"].values())[0],
                    periodicity,
                )
                idx_b = _find_nearest_grid_point(
                    xx, yy,
                    list(ref_system["minima"].values())[1],
                    periodicity,
                )

                # idx_a = _find_nearest_grid_point(
                #     xx, yy,
                #     (float(minima_df.iloc[0][cv_names[0]]), float(minima_df.iloc[0][cv_names[1]])),
                #     periodicity,
                # )
                # idx_b = _find_nearest_grid_point(
                #     xx, yy,
                #     (float(minima_df.iloc[1][cv_names[0]]), float(minima_df.iloc[1][cv_names[1]])),
                #     periodicity,
                # )
                _, _, mep_path = _compute_mep(zz, idx_a, idx_b, periodicity)
                rc_file = SAVE_PLOT.replace(".png", "_rc.png")
                plot_reaction_coordinate(xx, yy, zz, df, cv_names, periodicity,
                                         save_file=rc_file, mep_path=mep_path)
        plot_pmf(xx, yy, zz, periodicity, cv_names, df, e_max, mep_path=mep_path)
        # Save the MEP path as a separate CSV for potential further analysis
        if mep_path is not None:
            mep_df = pd.DataFrame({
                cv_names[0]: [float(xx[p]) for p in mep_path],
                cv_names[1]: [float(yy[p]) for p in mep_path],
                "energy_kcal_mol": [float(zz[p]) for p in mep_path],
            })
            mep_df.to_csv("mep_path.csv", index=False)
            print("MEP path saved to mep_path.csv")


if __name__ == "__main__":
    main()
