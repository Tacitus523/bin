#!/usr/bin/env python3

import argparse
import glob
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import re

# Constants
DPI = 150

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot walker completion as percentage of expected HILLS lines"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Subparser for analyzing a single folder
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze a single simulation folder"
    )
    analyze_parser.add_argument(
        "folder",
        type=str,
        help="Simulation folder to analyze"
    )
    analyze_parser.add_argument(
        "-e", "--expected",
        type=int,
        required=True,
        help="Expected number of lines in HILLS files"
    )
    analyze_parser.add_argument(
        "-o", "--output",
        type=str,
        default="walker_completion.png",
        help="Output filename (default: walker_completion.png)"
    )
    analyze_parser.add_argument(
        "-c", "--csv",
        type=str,
        default="walker_completion.csv",
        help="Output CSV filename (default: walker_completion.csv)"
    )
    
    # Subparser for comparing multiple CSV files
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare walker completion from multiple CSV files"
    )
    compare_parser.add_argument(
        "csvs",
        type=str,
        nargs="+",
        help="CSV files to compare"
    )
    compare_parser.add_argument(
        "-m", "--model-labels",
        type=str,
        nargs="+",
        help="Model labels for each CSV file (optional, uses filenames if not provided)"
    )
    compare_parser.add_argument(
        "-e", "--env-labels",
        type=str,
        nargs="+",
        help="Environment labels for each CSV file (optional)"
    )
    compare_parser.add_argument(
        "-o", "--output",
        type=str,
        default="walker_completion_comparison.png",
        help="Output filename (default: walker_completion_comparison.png)"
    )
    
    args = parser.parse_args()
    
    # Set default command if none provided
    if args.command is None:
        parser.print_help()
        exit(1)
    
    return args

def count_lines(filepath: Path) -> int:
    """Count lines in a file."""
    try:
        with open(filepath, "r") as f:
            return sum(1 for _ in f)
    except Exception as e:
        print(f"Warning: Could not read {filepath}: {e}")
        return 0

def analyze_folder(folder_path: Path) -> Dict[str, int]:
    """Analyze a folder and count lines in all HILLS.* files."""
    hills_files = glob.glob(str(folder_path / "HILLS.*"))
    hills_files.sort(key=lambda x: int(x.split(".")[-1]) if x.split(".")[-1].isdigit() else 0)
    
    line_counts = {}
    for hills_file in hills_files:
        filepath = Path(hills_file)
        
        # Extract walker number from HILLS.X filename
        match = re.search(r"HILLS\.(\d+)", filepath.name)
        if match:
            walker_num = int(match.group(1))
            walker_name = f"Walker {walker_num}"
        else:
            walker_name = filepath.name
        
        count = count_lines(filepath)
        line_counts[walker_name] = count
    
    return line_counts

def analyze_command(args: argparse.Namespace) -> None:
    """Handle the analyze command."""
    # Analyze the folder
    folder_path = Path(args.folder)
    if not folder_path.exists():
        print(f"Error: Folder {args.folder} does not exist")
        return
    
    line_counts = analyze_folder(folder_path)
    
    if not line_counts:
        print(f"Error: No HILLS.* files found in {args.folder}")
        return
    
    # Calculate percentages and prepare data
    data = []
    print(f"\nAnalyzing {args.folder}:")
    for walker_name, count in line_counts.items():
        percentage = (count / args.expected) * 100
        data.append({
            "walker": walker_name,
            "lines": count,
            "percentage": percentage
        })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(args.csv, index=False)
    print(f"Mean completion: {df['percentage'].mean():.2f}%")
    print(f"\nSaved completion data to: {args.csv}")
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(max(10, len(data) * 0.8), 8))
    
    # Color bars based on completion percentage using Greens colormap
    cmap = plt.cm.YlGn
    norm = plt.Normalize(vmin=0, vmax=100)
    palette = [cmap(norm(pct)) for pct in df["percentage"]]
    
    # Use seaborn barplot
    sns.barplot(
        data=df,
        x="walker",
        y="percentage",
        palette=palette,
        edgecolor="black",
        hue="walker",
        linewidth=1.5,
        ax=ax,
        legend=False
    )
    
    # Add reference line at 100%
    ax.axhline(y=100, color="red", linestyle="--", linewidth=2, alpha=0.7)
    
    # Formatting
    ax.set_xlabel("Walker")
    ax.set_ylabel("Completion (%)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    
    # Set y-axis to start at 0 and add some headroom
    max_pct = df["percentage"].max() if len(df) > 0 else 100
    ax.set_ylim(0, max(110, max_pct * 1.1))
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=DPI, bbox_inches="tight")
    print(f"Saved plot to: {args.output}")
    plt.close()

def compare_command(args: argparse.Namespace) -> None:
    """Handle the compare command."""
    # Load all CSV files
    dfs = []
    
    # Generate labels if not provided
    if args.model_labels is None:
        model_labels = [Path(csv).stem for csv in args.csvs]
    else:
        model_labels = args.model_labels
        if len(model_labels) != len(args.csvs):
            print(f"Error: Number of model labels ({len(model_labels)}) must match number of CSV files ({len(args.csvs)})")
            return
    
    if args.env_labels is not None and len(args.env_labels) != len(args.csvs):
        print(f"Error: Number of environment labels ({len(args.env_labels)}) must match number of CSV files ({len(args.csvs)})")
        return
    
    # Load CSVs and add labels
    for i, csv_file in enumerate(args.csvs):
        try:
            df = pd.read_csv(csv_file)
            df["model"] = model_labels[i]
            if args.env_labels is not None:
                df["environment"] = args.env_labels[i]
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not read {csv_file}: {e}")
    
    if not dfs:
        print("Error: No valid CSV files could be loaded")
        return
    
    # Combine all data
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Calculate mean completion per model
    model_completion = combined_df.groupby("model")["percentage"].mean().reset_index()
    model_completion = model_completion.sort_values("percentage", ascending=False)

    combined_df["model"] = pd.Categorical(combined_df["model"], ordered=True, categories=model_completion["model"])
    
    print("\nMean completion by model:")
    for _, row in model_completion.iterrows():
        print(f"  {row['model']}: {row['percentage']:.2f}%")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(max(10, len(model_labels) * 1.5), 8))
    
    # Determine if we have multiple environments
    has_multiple_envs = args.env_labels is not None and len(set(args.env_labels)) > 1
    
    # Prepare barplot kwargs based on whether we have multiple environments
    barplot_kwargs = {
        "data": combined_df,
        "x": "model",
        "y": "percentage",
        "ax": ax,
        "errorbar": "sd",
        "edgecolor": "black",
        "linewidth": 1.5,
    }
    
    if has_multiple_envs:
        # Use environment as hue
        barplot_kwargs["hue"] = "environment"
    else:
        # Color by completion percentage using YlGn colormap
        cmap = plt.cm.YlGn
        norm = plt.Normalize(vmin=0, vmax=100)
        palette = [cmap(norm(pct)) for pct in model_completion["percentage"]]
        
        barplot_kwargs["palette"] = palette
        barplot_kwargs["hue"] = "model"
        barplot_kwargs["legend"] = False
    

    sns.barplot(**barplot_kwargs)
    
    if has_multiple_envs:
        ax.legend(title="Environment", bbox_to_anchor=(1.05, 1), loc="upper left")
    
    # Add reference line at 100%
    ax.axhline(y=100, color="red", linestyle="--", linewidth=2, alpha=0.7)
    
    # Formatting
    ax.set_xlabel("Model")
    ax.set_ylabel("Mean Completion (%)")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    
    # Set y-axis limits
    max_pct = combined_df["percentage"].max() if len(combined_df) > 0 else 100
    ax.set_ylim(0, max(110, max_pct * 1.1))
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=DPI, bbox_inches="tight")
    print(f"\nSaved comparison plot to: {args.output}")
    plt.close()

    model_completion.to_csv("mean_walker_completion_by_model.csv", index=False)

def main():
    args = parse_args()

    sns.set_context("talk")
    if args.command == "analyze":
        analyze_command(args)
    elif args.command == "compare":
        compare_command(args)

if __name__ == "__main__":
    main()
