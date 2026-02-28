#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import List, Tuple
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

FORCE_UNIT = "eV/Å"
DISTANCE_UNIT = "Å"
DEFAULT_BASE_DIR = Path.cwd()
SUMMARY_FILENAME = "force_differences_summary.csv"
OUTPUT_SUMMARY = "all_cutoff_force_differences_summary.csv"
OUTPUT_BOXPLOT = "boxplot_abs_force_difference_by_method_cutoff.png"
OUTPUT_BARPLOT = "barplot_mean_max_force_diff.png"

NM_TO_ANGSTROM = 10

PALETTE = sns.color_palette("tab10")
PALETTE.pop(3)  # Remove red color

# Silence seaborn UserWarning about palette length
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"The palette list has more values .* than needed .*",
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read force_differences_summary.csv from cutoff folders, build a combined "
            "summary dataframe, and create boxplot + mean barplot."
        )
    )
    parser.add_argument(
        "--base_dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help=f"Base directory containing *_cutoff folders (default: {DEFAULT_BASE_DIR}).",
    )
    parser.add_argument(
        "--summary_output",
        type=Path,
        default=Path(OUTPUT_SUMMARY),
        help=f"Combined summary CSV output path (default: {OUTPUT_SUMMARY}).",
    )
    parser.add_argument(
        "--boxplot_output",
        type=Path,
        default=Path(OUTPUT_BOXPLOT),
        help=f"Boxplot output path (default: {OUTPUT_BOXPLOT}).",
    )
    parser.add_argument(
        "--barplot_output",
        type=Path,
        default=Path(OUTPUT_BARPLOT),
        help=f"Mean barplot output path (default: {OUTPUT_BARPLOT}).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output figure DPI (default: 300).",
    )
    return parser.parse_args()


def parse_cutoff(folder_name: str) -> Tuple[float, str]:
    raw = folder_name.replace("_cutoff", "")
    numeric = float(raw.replace(",", "."))
    numeric *= NM_TO_ANGSTROM
    label = f"{numeric:.1f}"
    return numeric, label


def load_summary_dataframe(base_dir: Path) -> pd.DataFrame:
    if not base_dir.is_dir():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    frames: List[pd.DataFrame] = []
    cutoff_levels: List[Tuple[float, str]] = []

    for cutoff_dir in sorted(base_dir.glob("*_cutoff")):
        csv_path = cutoff_dir / SUMMARY_FILENAME
        if not csv_path.is_file():
            continue

        cutoff_value, cutoff_label = parse_cutoff(cutoff_dir.name)
        cutoff_levels.append((cutoff_value, cutoff_label))

        frame = pd.read_csv(csv_path)
        required_columns = {"Method", "Abs. Force Difference"}
        missing = required_columns.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing columns {missing} in {csv_path}")

        frame = frame.copy()
        frame["Cutoff"] = cutoff_label
        frame["CutoffValue"] = cutoff_value
        frame["SourceFile"] = str(csv_path)
        frames.append(frame)

    if not frames:
        raise ValueError(f"No {SUMMARY_FILENAME} found in {base_dir}/*_cutoff")

    summary_df = pd.concat(frames, ignore_index=True)
    ordered_cutoffs = [label for _, label in sorted(set(cutoff_levels), key=lambda item: item[0])]
    summary_df["Cutoff"] = pd.Categorical(summary_df["Cutoff"], categories=ordered_cutoffs, ordered=True)
    return summary_df


def get_method_order_by_median(summary_df: pd.DataFrame) -> List[str]:
    median_df = (
        summary_df.groupby("Method", observed=True)["Abs. Force Difference"]
        .median()
        .sort_values(ascending=False)
    )
    return median_df.index.tolist()


def plot_boxplot(summary_df: pd.DataFrame, output_path: Path, dpi: int) -> None:
    method_order = get_method_order_by_median(summary_df)
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.boxplot(
        data=summary_df,
        x="Method",
        y="Abs. Force Difference",
        hue="Cutoff",
        palette=PALETTE,
        order=method_order,
        showfliers=False,
        ax=ax
    )
    ax.set_xlabel("Method")
    ax.set_ylabel(f"Abs. Force Difference ({FORCE_UNIT})")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(title=f"Cutoff ({DISTANCE_UNIT})", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_mean_barplot(summary_df: pd.DataFrame, output_path: Path, dpi: int) -> None:
    mean_df = (
        summary_df.groupby(["Cutoff", "Method"], observed=True, as_index=False)["Abs. Force Difference"]
        .mean()
        .rename(columns={"Abs. Force Difference": "Mean_Max_Abs_Force_Diff"})
    )
    method_order = get_method_order_by_median(summary_df)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=mean_df,
        x="Cutoff",
        y="Mean_Max_Abs_Force_Diff",
        hue="Method",
        hue_order=method_order,
        palette=PALETTE,
        ax=ax,
    )
    ax.set_xlabel(f"Cutoff ({DISTANCE_UNIT})")
    ax.set_ylabel(f"Mean Max. |Δ Force| ({FORCE_UNIT})")
    ax.grid(alpha=0.3, axis="y")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=10)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    sns.set_context("talk")

    summary_df = load_summary_dataframe(args.base_dir)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.summary_output, index=False)

    plot_boxplot(summary_df, args.boxplot_output, args.dpi)
    plot_mean_barplot(summary_df, args.barplot_output, args.dpi)

    print(f"Loaded {len(summary_df)} rows from cutoff folders in {args.base_dir}")
    print(f"Saved combined summary CSV to {args.summary_output.resolve()}")
    print(f"Saved boxplot to {args.boxplot_output.resolve()}")
    print(f"Saved mean barplot to {args.barplot_output.resolve()}")


if __name__ == "__main__":
    main()

