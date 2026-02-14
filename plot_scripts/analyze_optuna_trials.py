#!/usr/bin/env python3
"""Analyze Optuna trials from journal log files and create visualizations.

Reads Optuna JournalFileStorage logs (optuna_trials.log), extracts trial data
including hyperparameters, objective values, timing, and intermediate values,
then creates ranking plots, hyperparameter analysis plots, and duration plots.

Supports multiple studies from subdirectories (analogous to Keras Tuner projects).

Usage examples:
    # Analyze all studies in the current directory
    python analyze_optuna_trials.py -t .

    # Analyze specific studies
    python analyze_optuna_trials.py -t /path/to/grid_search -s 01_num_radial_basis 02_num_cutoff_basis

    # Analyze a single log file
    python analyze_optuna_trials.py -l optuna_trials.log

    # Hyperband-style multi-parameter analysis
    python analyze_optuna_trials.py -t . --hyperband
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch


ENERGY_UNIT = "eV"
FORCE_UNIT = r"$\frac{\mathrm{eV}}{\mathrm{\AA}}$"

SCORE_LABEL = f"Force RMSE ({FORCE_UNIT})"
COLORMAP_NAME_NUMERICAL = "YlOrRd"
COLORMAP_NAME_CATEGORICAL = "tab10"
SINGLE_FIG_SIZE = (12, 6)
MULTIPLOT_FIG_SIZE = (5, 4)
DPI = 150

# Optuna journal op_codes
OP_CREATE_STUDY = 0
OP_TRIAL_START = 4
OP_SET_PARAM = 5
OP_TRIAL_COMPLETE = 6
OP_INTERMEDIATE_VALUE = 7
OP_SET_SYSTEM_ATTR = 9


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze Optuna trials from journal log files and create visualizations"
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "-t",
        "--trial-dir",
        type=str,
        default=None,
        help="Parent directory containing study subdirectories with optuna_trials.log files",
    )
    input_group.add_argument(
        "-l",
        "--log-files",
        nargs="+",
        type=str,
        default=None,
        help="Direct path(s) to Optuna journal log file(s)",
    )
    parser.add_argument(
        "-s",
        "--study-names",
        nargs="+",
        type=str,
        default=None,
        help="Specific study directory names to analyze (default: all)",
    )
    parser.add_argument(
        "-f",
        "--log-filename",
        type=str,
        default="optuna_trials.log",
        help="Name of log file in each study directory (default: optuna_trials.log)",
    )
    parser.add_argument(
        "-n",
        "--n-best",
        type=int,
        default=12,
        help="Number of best trials to analyze (default: 12)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="trial_analysis",
        help="Output directory for CSV and plots (default: trial_analysis)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="score",
        help="Metric column to use for ranking (default: score)",
    )
    parser.add_argument(
        "--minimize",
        action="store_true",
        default=True,
        help="Whether to minimize the metric (default: True)",
    )
    parser.add_argument(
        "--maximize",
        action="store_false",
        dest="minimize",
        help="Whether to maximize the metric",
    )
    parser.add_argument(
        "--grid_search",
        action="store_true",
        default=True,
        help="Whether the tuner used grid search (default: True)",
    )
    parser.add_argument(
        "--hyperband",
        action="store_false",
        dest="grid_search",
        help="Whether the tuner used Hyperband / multi-parameter search",
    )
    args = parser.parse_args()

    # Default to current directory if nothing specified
    if args.trial_dir is None and args.log_files is None:
        args.trial_dir = "."

    if args.trial_dir is not None:
        args.trial_dir = os.path.abspath(args.trial_dir)
        if not os.path.exists(args.trial_dir):
            raise FileNotFoundError(f"Trial directory not found: {args.trial_dir}")

    return args


# ---------------------------------------------------------------------------
# Optuna Journal Log Parsing
# ---------------------------------------------------------------------------


def _decode_param_value(
    param_value_internal: Any, distribution_str: str
) -> Any:
    """Decode an Optuna internal parameter value using its distribution.

    For CategoricalDistribution the internal value is an index into choices.
    For numeric distributions the internal value is the actual value.
    """
    try:
        dist = json.loads(distribution_str)
    except (json.JSONDecodeError, TypeError):
        return param_value_internal

    dist_name = dist.get("name", "")
    attrs = dist.get("attributes", {})

    if dist_name == "CategoricalDistribution":
        choices = attrs.get("choices", [])
        idx = int(param_value_internal)
        if 0 <= idx < len(choices):
            return choices[idx]
        return param_value_internal

    # IntDistribution → cast to int
    if dist_name == "IntDistribution":
        return int(param_value_internal)

    # FloatDistribution → keep as float
    return param_value_internal


def parse_optuna_journal_log(log_path: str) -> Dict[str, Any]:
    """Parse an Optuna JournalFileStorage log and return structured data.

    Returns a dict with keys:
        study_name, directions,
        trials: {trial_id: {params, values, state, datetime_start,
                            datetime_complete, intermediate_values,
                            system_attrs, distributions}}
    """
    study_name: Optional[str] = None
    directions: List[int] = []
    trials: Dict[int, Dict[str, Any]] = {}
    # Map worker_id → datetime_start for the next trial created by that worker
    pending_starts: Dict[str, str] = {}

    with open(log_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            op = rec.get("op_code")
            worker_id = rec.get("worker_id", "")

            if op == OP_CREATE_STUDY:
                study_name = rec.get("study_name", study_name)
                directions = rec.get("directions", directions)

            elif op == OP_TRIAL_START:
                # OP_TRIAL_START does NOT contain trial_id, only study_id.
                # Store by worker_id so the next OP_SET_PARAM from the same
                # worker can claim it.
                dt_start = rec.get("datetime_start")
                if dt_start is not None:
                    pending_starts[worker_id] = dt_start

            elif op == OP_SET_PARAM:
                tid = rec["trial_id"]
                if tid not in trials:
                    trials[tid] = {
                        "params": {},
                        "distributions": {},
                        "values": [],
                        "state": None,
                        "datetime_start": None,
                        "datetime_complete": None,
                        "intermediate_values": {},
                        "system_attrs": {},
                    }
                # Assign pending start time from the same worker
                if trials[tid]["datetime_start"] is None and worker_id in pending_starts:
                    trials[tid]["datetime_start"] = pending_starts.pop(worker_id)

                pname = rec["param_name"]
                pval_internal = rec["param_value_internal"]
                dist_str = rec.get("distribution", "")
                trials[tid]["params"][pname] = _decode_param_value(pval_internal, dist_str)
                trials[tid]["distributions"][pname] = dist_str

            elif op == OP_SET_SYSTEM_ATTR:
                tid = rec["trial_id"]
                if tid not in trials:
                    trials[tid] = {
                        "params": {},
                        "distributions": {},
                        "values": [],
                        "state": None,
                        "datetime_start": None,
                        "datetime_complete": None,
                        "intermediate_values": {},
                        "system_attrs": {},
                    }
                sys_attr = rec.get("system_attr", {})
                trials[tid]["system_attrs"].update(sys_attr)

            elif op == OP_INTERMEDIATE_VALUE:
                tid = rec["trial_id"]
                if tid not in trials:
                    trials[tid] = {
                        "params": {},
                        "distributions": {},
                        "values": [],
                        "state": None,
                        "datetime_start": None,
                        "datetime_complete": None,
                        "intermediate_values": {},
                        "system_attrs": {},
                    }
                step = rec["step"]
                trials[tid]["intermediate_values"][step] = rec["intermediate_value"]

            elif op == OP_TRIAL_COMPLETE:
                tid = rec["trial_id"]
                if tid not in trials:
                    trials[tid] = {
                        "params": {},
                        "distributions": {},
                        "values": [],
                        "state": None,
                        "datetime_start": None,
                        "datetime_complete": None,
                        "intermediate_values": {},
                        "system_attrs": {},
                    }
                trials[tid]["state"] = rec.get("state")
                trials[tid]["values"] = rec.get("values", [])
                trials[tid]["datetime_complete"] = rec.get("datetime_complete")

    return {
        "study_name": study_name or "unknown",
        "directions": directions,
        "trials": trials,
    }


def load_optuna_trials(log_path: str, study_label: str, is_grid_search: bool) -> Optional[pd.DataFrame]:
    """Load an Optuna journal log and convert to a DataFrame.

    Parameters
    ----------
    log_path : str
        Path to the optuna_trials.log file.
    study_label : str
        Label used as ``project_name`` in the output (typically the directory name).

    Returns
    -------
    pd.DataFrame or None
    """
    if not os.path.exists(log_path):
        print(f"Warning: Log file not found: {log_path}")
        return None

    parsed = parse_optuna_journal_log(log_path)
    trials = parsed["trials"]

    if not trials:
        print(f"Warning: No trials found in {log_path}")
        return None

    state_map = {1: "COMPLETE", 2: "PRUNED", 3: "FAIL", 4: "WAITING", 5: "RUNNING"}

    rows: List[Dict[str, Any]] = []
    for tid, tdata in sorted(trials.items()):
        state_code = tdata["state"]
        status = state_map.get(state_code, f"UNKNOWN({state_code})")

        # Primary objective value (first element of values list)
        score = tdata["values"][0] if tdata["values"] else np.nan

        row: Dict[str, Any] = {
            "project_name": study_label,
            "trial_id": str(tid),
            "status": status,
            "score": score,
        }

        # Hyperparameters
        for pname, pval in tdata["params"].items():
            row[f"param_{pname}"] = pval

        # Duration from timestamps
        dt_start_str = tdata.get("datetime_start")
        dt_end_str = tdata.get("datetime_complete")
        if dt_start_str and dt_end_str:
            try:
                dt_start = datetime.fromisoformat(dt_start_str)
                dt_end = datetime.fromisoformat(dt_end_str)
                duration_minutes = (dt_end - dt_start).total_seconds() / 60.0
                row["duration"] = duration_minutes
            except (ValueError, TypeError):
                row["duration"] = np.nan
        else:
            row["duration"] = np.nan

        rows.append(row)

    if not rows:
        print(f"Warning: No trial data extracted from {log_path}")
        return None

    df = pd.DataFrame(rows)
    # Only keep completed trials
    df = df[df["status"] == "COMPLETE"].reset_index(drop=True)

    # Combine trials with identical parameter values (e.g. duplicate grid_id)
    param_cols = [col for col in df.columns if col.startswith("param_")]
    if param_cols and len(df) > 0:
        group_cols = ["project_name"] + param_cols
        # Convert param columns to string for grouping (handles lists/arrays)
        str_cols = {col: df[col].astype(str) for col in param_cols}
        df_groupkey = df.assign(**str_cols)
        grouped = df_groupkey.groupby(group_cols, dropna=False)
        if len(grouped) < len(df):
            agg_dict: Dict[str, Any] = {
                "trial_id": lambda x: "/".join(x.astype(str)),
                "status": "first",
                "score": "mean",
            }
            if "duration" in df.columns:
                agg_dict["duration"] = "mean"
            df = grouped.agg(agg_dict).reset_index()
            # Restore original param dtypes
            for col in param_cols:
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    pass

    # Order trials by original search space order from the logs
    if param_cols and len(df) > 0 and is_grid_search:
        search_space: Dict[str, list] = {}
        for tdata in parsed["trials"].values():
            ss = tdata.get("system_attrs", {}).get("search_space")
            if ss:
                search_space = ss
                break

        if search_space:
            sort_keys = []
            for col in param_cols:
                pname = col.replace("param_", "")
                if pname in search_space:
                    order = search_space[pname]
                    pos_map = {v: i for i, v in enumerate(order)}
                    key_col = f"_sort_{col}"
                    df[key_col] = df[col].map(pos_map)
                    sort_keys.append(key_col)
            if sort_keys:
                df = df.sort_values(sort_keys).reset_index(drop=True)
                df = df.drop(columns=sort_keys)

        # Reassign trial_id to reflect order
        df["trial_id"] = df.index.astype(str)

    return df


# ---------------------------------------------------------------------------
# Data Wrangling (reused from Keras Tuner version)
# ---------------------------------------------------------------------------


def get_best_trials(
    trials_df: pd.DataFrame,
    metric: str,
    n_best: int,
    minimize: bool = True,
) -> pd.DataFrame:
    """Get the best n trials based on the specified metric."""
    metric_col = None
    if metric in trials_df.columns:
        metric_col = metric
    elif f"metric_{metric}" in trials_df.columns:
        metric_col = f"metric_{metric}"
    elif "score" in trials_df.columns:
        metric_col = "score"
        print(f"Warning: Metric '{metric}' not found, using 'score' instead")
    else:
        raise ValueError(
            f"Metric '{metric}' not found in trial data. "
            f"Available columns: {list(trials_df.columns)}"
        )

    df_sorted = trials_df.sort_values(metric_col, ascending=minimize)
    best_trials = df_sorted.head(n_best).copy()
    best_trials["rank"] = range(1, len(best_trials) + 1)
    best_trials = best_trials.sort_values(["project_name", "trial_id"]).reset_index(drop=True)
    return best_trials


def clean_trials(trials_df: pd.DataFrame) -> pd.DataFrame:
    """Clean the trials DataFrame (layer neuron cleanup, dtype fixes)."""
    layer_columns = [col for col in trials_df.columns if "_n_layers" in col]

    def layer_cleanup(data_point: pd.Series) -> pd.Series:
        for layer_col in layer_columns:
            prefix = layer_col.replace("_n_layers", "")
            neuron_columns = [
                col
                for col in trials_df.columns
                if col.startswith(prefix) and "_neurons_" in col
            ]
            if layer_col in data_point:
                n_layers = data_point[layer_col]
                if pd.isna(n_layers):
                    continue
                for neuron_col in neuron_columns:
                    layer_index = int(neuron_col.split("_neurons_")[-1])
                    if layer_index >= n_layers:
                        data_point[neuron_col] = np.nan
        return data_point

    trials_df = trials_df.copy().apply(layer_cleanup, axis=1)

    neuron_columns = [col for col in trials_df.columns if "_neurons_" in col]
    for col in neuron_columns:
        trials_df[col] = trials_df[col].astype("Int64")

    return trials_df


def categorize_parameters(
    best_trials_df: pd.DataFrame, all_trials_df: pd.DataFrame
) -> pd.DataFrame:
    """Convert parameter columns to categorical with all possible values from all trials."""
    best_trials_df = best_trials_df.copy()
    param_cols = [col for col in best_trials_df.columns if col.startswith("param_")]

    for col in param_cols:
        all_categories = all_trials_df[col].dropna().unique()
        best_trials_df[col] = pd.Categorical(
            best_trials_df[col],
            categories=sorted(all_categories),
            ordered=True,
        )

    return best_trials_df


def save_trials_csv(trials_df: pd.DataFrame, output_path: str) -> None:
    """Save trials data to CSV file."""
    trials_df.to_csv(output_path, index=False)
    print(f"Saved best trials data to: {output_path}")


# ---------------------------------------------------------------------------
# Plotting Helpers
# ---------------------------------------------------------------------------


def create_trial_identifiers(
    top_rank_df: pd.DataFrame, verbose: bool = False
) -> pd.DataFrame:
    """Create trial identifiers combining project name and trial ID."""
    identifiers = []
    for _, row in top_rank_df.iterrows():
        clean_project_name = row["project_name"].replace("_", " ").title()
        clean_trial_id = row["trial_id"]
        if verbose:
            identifier = f"{clean_project_name}\n{clean_trial_id}"
        else:
            identifier = clean_trial_id
        identifiers.append(identifier)

    top_rank_df = top_rank_df.copy()
    top_rank_df["identifier"] = identifiers
    return top_rank_df


def get_colors(
    max_value: float, values: List[float]
) -> List[Tuple[float, float, float, float]]:
    """Generate a list of colors from the colormap based on max value."""
    norm = plt.Normalize(vmin=0, vmax=max_value)
    cmap = sns.color_palette(COLORMAP_NAME_NUMERICAL, as_cmap=True)
    return [cmap(norm(v)) for v in values]


def _make_axes(
    n: int, single_figsize: Tuple = SINGLE_FIG_SIZE, multi_figsize: Tuple = MULTIPLOT_FIG_SIZE
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Create a grid of subplots, returning (fig, flat list of axes)."""
    n_cols = min(3, n)
    n_rows = (n + n_cols - 1) // n_cols
    if n == 1:
        fig, ax = plt.subplots(1, 1, figsize=single_figsize)
        return fig, [ax]
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(multi_figsize[0] * n_cols, multi_figsize[1] * n_rows)
    )
    if n_rows == 1:
        axes_list = list(axes) if n > 1 else [axes]
    else:
        axes_list = list(axes.flatten())
    return fig, axes_list


# ---------------------------------------------------------------------------
# Plot Functions
# ---------------------------------------------------------------------------


def plot_grid_search_analysis(
    trials_df: pd.DataFrame,
    metric_col: str,
    output_dir: str,
    target_project_name: Optional[str] = None,
) -> None:
    """Create bar plots for hyperparameter analysis."""
    metric_max = trials_df[metric_col].max()
    color_max = metric_max

    if target_project_name is not None:
        trials_df = trials_df[trials_df["project_name"] == target_project_name]
        if len(trials_df) == 0:
            print(f"Warning: No trials found for project {target_project_name}")
            return
        metric_max = trials_df[metric_col].max()

    trials_df = trials_df.dropna(how="all", axis=1)
    trials_df = trials_df.sort_values(["project_name", "trial_id"])

    project_names = trials_df["project_name"].unique().tolist()
    n_projects = len(project_names)
    fig, axes = _make_axes(n_projects)

    for i, project_name in enumerate(project_names):
        ax = axes[i] if i < len(axes) else None
        if ax is None:
            continue

        project_df = trials_df[trials_df["project_name"] == project_name]
        param_cols = [col for col in project_df.columns if col.startswith("param_")]
        param_df = project_df[param_cols].dropna(how="all", axis=1)
        clean_project_name = project_name.replace("_", " ").title()

        do_legend = target_project_name is not None

        colors = get_colors(color_max, project_df[metric_col])
        sns.barplot(
            x="trial_id",
            y=metric_col,
            data=project_df,
            ax=ax,
            palette=colors,
            hue="trial_id",
            legend=do_legend,
        )
        ax.set_title(clean_project_name if target_project_name is None else None)
        ax.set_xlabel("Trial ID")
        ax.set_ylabel(SCORE_LABEL)
        ax.set_ylim(0, metric_max * 1.1)

        if not do_legend:
            continue

        # Build legend with parameter values
        handles, labels = ax.get_legend_handles_labels()
        new_handles, new_labels = [], []
        for j, (handle, label) in enumerate(zip(handles, labels)):
            present_params = param_df.iloc[j].dropna()
            for column_name in present_params.index:
                if "neuron" in column_name:
                    present_params = present_params.astype("Int64")

            clean_parameter_names = [
                p.replace("param_", "").replace("_", " ").title()
                for p in present_params.index
            ]
            label_text = "\n".join(
                [f"{present_params.iloc[k]}" for k, _ in enumerate(clean_parameter_names)]
            )
            new_label = rf"$\mathbf{{Trial~{label}:}}$" + f"\n{label_text}"
            if len(param_df.columns) > 1:
                label_text = (
                    "["
                    + ",".join(
                        [
                            f"{present_params.iloc[k]}"
                            for k, col in enumerate(present_params.index)
                            if "layer" not in col
                        ]
                    )
                    + "]"
                )
                new_label = rf"$\mathbf{{Trial~{label}:}}$" + f"{label_text}"
            new_handles.append(Patch(facecolor="none", edgecolor="none"))
            new_labels.append(new_label)

        ax.legend(
            new_handles,
            new_labels,
            title=clean_project_name,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            handlelength=0,
            handletextpad=0.3,
            borderpad=0.5,
        )

    for i in range(n_projects, len(axes)):
        axes[i].set_visible(False)

    if not (target_project_name is not None):
        plt.tight_layout()

    filename = (
        f"hyperparameter_analysis_{project_names[-1]}.png"
        if target_project_name
        else "hyperparameter_analysis.png"
    )
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path, dpi=DPI, bbox_inches="tight")
    plt.close()


def plot_trial_ranking(
    trials_df: pd.DataFrame,
    metric_col: str,
    output_dir: str,
    target_project_name: Optional[str] = None,
) -> None:
    """Create a bar plot showing trial rankings."""
    max_metric = trials_df[metric_col].max()

    if target_project_name is not None:
        trials_df = trials_df[trials_df["project_name"] == target_project_name]
    project_names = trials_df["project_name"].unique().tolist()
    n_projects = len(project_names)
    verbose = target_project_name is None and n_projects > 1

    top_rank_df = trials_df.sort_values(metric_col).reset_index(drop=True)

    plt.figure(figsize=SINGLE_FIG_SIZE)

    colors = get_colors(max_metric, top_rank_df[metric_col])
    top_rank_df = create_trial_identifiers(top_rank_df, verbose=verbose)

    ax = sns.barplot(
        x="identifier",
        y=metric_col,
        data=top_rank_df,
        palette=colors,
        hue="identifier",
        legend=False,
    )

    plt.xlabel("Trial ID")
    plt.ylabel(SCORE_LABEL)

    if verbose:
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30)

    for i, (_, row) in enumerate(top_rank_df.iterrows()):
        value = row[metric_col]
        ax.text(i, value, f"{value:.4f}", ha="center", va="bottom")

    plt.tight_layout()

    filename = (
        f"trial_ranking_{target_project_name}.png"
        if target_project_name
        else "trial_ranking.png"
    )
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path, dpi=DPI, bbox_inches="tight")
    plt.close()


def plot_duration_analysis(
    trials_df: pd.DataFrame,
    metric_col: str,
    output_dir: str,
    target_project_name: Optional[str] = None,
    do_sort: bool = False,
) -> None:
    """Create plots analyzing trial duration vs performance."""
    max_metric = trials_df[metric_col].max()
    trials_df = trials_df.copy()

    if target_project_name is not None:
        trials_df = trials_df[trials_df["project_name"] == target_project_name]
        if len(trials_df) == 0:
            print(f"Warning: No duration data found for project {target_project_name}")
            return

    if do_sort:
        trials_df = trials_df.sort_values(
            [metric_col, "project_name", "trial_id"], ascending=[True, True, True]
        )
    else:
        trials_df = trials_df.sort_values(
            ["project_name", "trial_id"], ascending=[True, True]
        )

    if "duration" not in trials_df.columns or trials_df["duration"].isna().all():
        print("Warning: No timing information found in trials data")
        return

    trials_df = trials_df.dropna(subset=["duration"])
    if len(trials_df) == 0:
        print("Warning: No valid duration data found")
        return

    project_names = trials_df["project_name"].unique().tolist()
    n_projects = len(project_names)
    do_legend = not (target_project_name is not None or n_projects == 1)

    fig, axes = _make_axes(n_projects)

    for i, project_name in enumerate(project_names):
        project_df = trials_df[trials_df["project_name"] == project_name]
        clean_project_name = project_name.replace("_", " ").title()
        ax = axes[i] if i < len(axes) else None
        if ax is None:
            continue

        project_df = create_trial_identifiers(project_df, verbose=False)
        colors = get_colors(max_metric, project_df[metric_col])
        sns.barplot(
            x="identifier",
            y="duration",
            data=project_df,
            palette=colors,
            hue="identifier",
            legend=False,
            ax=ax,
        )
        ax.set_title(clean_project_name if do_legend else None)
        ax.set_xlabel("Trial ID")
        ax.set_ylabel("Duration (minutes)")

    for i in range(n_projects, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    filename = (
        f"duration_analysis_{target_project_name}.png"
        if target_project_name
        else "duration_analysis.png"
    )
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path, dpi=DPI, bbox_inches="tight")
    plt.close()

    if target_project_name is None:
        print(f"\nDuration Statistics:")
        print(f"Mean duration: {trials_df['duration'].mean():.1f} minutes")
        print(f"Median duration: {trials_df['duration'].median():.1f} minutes")
        print(f"Min duration: {trials_df['duration'].min():.1f} minutes")
        print(f"Max duration: {trials_df['duration'].max():.1f} minutes")


def plot_hyperband_analysis(
    trials_df: pd.DataFrame,
    metric_col: str,
    output_dir: str,
    target_param_name: Optional[str] = None,
) -> None:
    """Create multi-parameter analysis plots (swarm plots per hyperparameter)."""
    do_legend = target_param_name is not None

    hyperparam_column_names: List[str] = []
    for col in trials_df.columns:
        if target_param_name is not None:
            hyperparam_column_names = [target_param_name]
            break
        if not col.startswith("param_"):
            continue
        if isinstance(trials_df[col].dtype, pd.CategoricalDtype):
            if len(trials_df[col].cat.categories) < 2:
                continue
        elif trials_df[col].nunique() < 2:
            continue
        hyperparam_column_names.append(col)

    if not hyperparam_column_names:
        print("Warning: No hyperparameter columns with >1 unique value found")
        return

    n_params = len(hyperparam_column_names)
    fig, axes = _make_axes(n_params)

    for i, column_name in enumerate(hyperparam_column_names):
        ax = axes[i] if i < len(axes) else None
        if ax is None:
            continue

        param_df = trials_df.copy().dropna(subset=[column_name])
        param_df = param_df.sort_values(column_name)
        clean_param_name = column_name.replace("param_", "").replace("_", " ").title()

        # All categories
        if isinstance(trials_df[column_name].dtype, pd.CategoricalDtype):
            all_categories = trials_df[column_name].cat.categories
        else:
            all_categories = sorted(trials_df[column_name].dropna().unique())

        param_id_mapping = {cat: idx for idx, cat in enumerate(all_categories)}
        column_id_mapping = {idx: cat for idx, cat in enumerate(all_categories)}
        param_df["param_id"] = param_df[column_name].map(param_id_mapping)
        x_order = list(range(len(all_categories)))

        do_label_xticks = all(len(str(c)) < 10 for c in all_categories)

        sns.swarmplot(
            x="param_id",
            y=metric_col,
            data=param_df,
            hue="param_id",
            order=x_order,
            palette=COLORMAP_NAME_CATEGORICAL,
            marker="x",
            linewidth=2,
            size=8,
            ax=ax,
            legend=do_legend,
        )
        ax.set_title(clean_param_name if not do_legend else None)
        ax.set_xlabel("Parameter ID")
        ax.set_ylabel(SCORE_LABEL)

        if do_label_xticks:
            ax.set_xlabel(clean_param_name)
            ax.set_xticks(x_order)
            ax.set_xticklabels([column_id_mapping[idx] for idx in x_order])

        if not do_legend:
            continue
        if do_label_xticks:
            continue

        # Legend mapping param_id → actual value
        handles, labels = ax.get_legend_handles_labels()
        new_handles, new_labels = [], []
        for handle, label in zip(handles, labels):
            new_label = f"{label}: {column_id_mapping[int(label)]}"
            handle.set_alpha(1)
            new_handles.append(handle)
            new_labels.append(new_label)

        ax.legend(
            new_handles,
            new_labels,
            title=clean_param_name,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )

    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)

    if not do_legend:
        plt.tight_layout()

    filename = (
        f"hyperparameter_analysis_{target_param_name.replace('param_', '')}.png"
        if target_param_name
        else "hyperparameter_analysis.png"
    )
    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path, dpi=DPI, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Main function."""
    args = parse_arguments()

    print("=" * 60)
    print("OPTUNA TRIAL ANALYSIS")
    print("=" * 60)
    print(f"Number of best trials: {args.n_best}")
    print(f"Target metric: {args.metric}")
    print(f"Optimization: {'minimize' if args.minimize else 'maximize'}")
    print(f"Output directory: {args.output_dir}")
    print()

    os.makedirs(args.output_dir, exist_ok=True)

    # Discover log files
    log_entries: List[Tuple[str, str]] = []  # (study_label, log_path)

    if args.log_files is not None:
        for lf in args.log_files:
            lf = os.path.abspath(lf)
            label = Path(lf).parent.name or Path(lf).stem
            log_entries.append((label, lf))
    else:
        trial_dir = args.trial_dir
        if args.study_names:
            candidate_dirs = args.study_names
        else:
            # Auto-discover: subdirectories containing the log file
            candidate_dirs = sorted(
                [
                    d
                    for d in os.listdir(trial_dir)
                    if os.path.isdir(os.path.join(trial_dir, d))
                    and os.path.exists(os.path.join(trial_dir, d, args.log_filename))
                ]
            )
            # Also check if log file exists in the trial_dir itself
            if os.path.exists(os.path.join(trial_dir, args.log_filename)) and not candidate_dirs:
                label = Path(trial_dir).name
                log_entries.append(
                    (label, os.path.join(trial_dir, args.log_filename))
                )

        for d in candidate_dirs:
            log_path = os.path.join(trial_dir, d, args.log_filename)
            if os.path.exists(log_path):
                log_entries.append((d, log_path))
            else:
                print(f"Warning: Log file not found: {log_path}")

    if not log_entries:
        print("Error: No Optuna log files found.")
        sys.exit(1)

    print(f"Found {len(log_entries)} study log file(s):")
    for label, path in log_entries:
        print(f"  {label}: {path}")
    print()

    # Load all trials
    all_trials: List[pd.DataFrame] = []
    best_trials: List[pd.DataFrame] = []

    for study_label, log_path in log_entries:
        print(f"Loading study: {study_label}")

        trials_data = load_optuna_trials(log_path, study_label, args.grid_search)
        if trials_data is None or len(trials_data) == 0:
            continue

        best_project_trials_df = get_best_trials(
            trials_data, args.metric, args.n_best, args.minimize
        )

        all_trials.append(trials_data)
        best_trials.append(best_project_trials_df)

    if not best_trials:
        print("Error: No trial data found across all studies.")
        sys.exit(1)

    print(f"\nCombining best trials from all studies...")
    print(
        f"Found total of {sum(len(df) for df in best_trials)} best trials "
        f"from {len(best_trials)} studies."
    )

    all_trials_df = pd.concat(all_trials, ignore_index=True)
    best_trials_df = pd.concat(best_trials, ignore_index=True)

    all_trials_df = clean_trials(all_trials_df)
    best_trials_df = clean_trials(best_trials_df)

    best_trials_df = categorize_parameters(best_trials_df, all_trials_df)

    # Determine metric column name
    metric_col = args.metric
    if args.metric not in best_trials_df.columns:
        if f"metric_{args.metric}" in best_trials_df.columns:
            metric_col = f"metric_{args.metric}"
        else:
            metric_col = "score"

    # Save to CSV
    csv_path = os.path.join(args.output_dir, "best_trials.csv")
    save_trials_csv(best_trials_df, csv_path)

    # Set style
    sns.set_context("talk")

    study_labels = [label for label, _ in log_entries]

    if args.grid_search:
        print("\nCreating comprehensive visualizations (all trials)...")
        plot_trial_ranking(best_trials_df, metric_col, args.output_dir)
        plot_grid_search_analysis(best_trials_df, metric_col, args.output_dir)
        plot_duration_analysis(best_trials_df, metric_col, args.output_dir)

        print("\nCreating study-specific visualizations...")
        for study_label in study_labels:
            plot_trial_ranking(
                best_trials_df, metric_col, args.output_dir, target_project_name=study_label
            )
            plot_grid_search_analysis(
                best_trials_df, metric_col, args.output_dir, target_project_name=study_label
            )
            plot_duration_analysis(
                best_trials_df, metric_col, args.output_dir, target_project_name=study_label
            )
    else:
        print("\nCreating Hyperband-specific visualizations...")
        plot_hyperband_analysis(best_trials_df, metric_col, args.output_dir)
        plot_trial_ranking(best_trials_df, metric_col, args.output_dir)
        plot_duration_analysis(best_trials_df, metric_col, args.output_dir, do_sort=True)

        print("\nCreating parameter-specific visualizations...")
        parameter_cols = [
            col for col in best_trials_df.columns if col.startswith("param_")
        ]
        for parameter_col in parameter_cols:
            plot_hyperband_analysis(
                best_trials_df, metric_col, args.output_dir, target_param_name=parameter_col
            )

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

    print(f"\nSummary Statistics for {metric_col}:")
    sorted_df = best_trials_df.sort_values(metric_col, ascending=True)
    print(f"Best value: {sorted_df[metric_col].iloc[0]:.6f}")
    print(f"Worst value: {sorted_df[metric_col].iloc[-1]:.6f}")
    print(f"Mean: {best_trials_df[metric_col].mean():.6f}")
    print(f"Std: {best_trials_df[metric_col].std():.6f}")
    print(f"Range: {best_trials_df[metric_col].max() - best_trials_df[metric_col].min():.6f}")


if __name__ == "__main__":
    main()
