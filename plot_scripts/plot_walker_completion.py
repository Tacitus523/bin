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
    parser.add_argument(
        "folder",
        type=str,
        help="Simulation folder to analyze"
    )
    parser.add_argument(
        "-e", "--expected",
        type=int,
        required=True,
        help="Expected number of lines in HILLS files"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="walker_completion.png",
        help="Output filename (default: walker_completion.png)"
    )
    parser.add_argument(
        "-c", "--csv",
        type=str,
        default="walker_completion.csv",
        help="Output CSV filename (default: walker_completion.csv)"
    )
    return parser.parse_args()

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
    hills_files.sort(key=lambda x: int(x.split('.')[-1]) if x.split('.')[-1].isdigit() else 0)
    
    line_counts = {}
    for hills_file in hills_files:
        filepath = Path(hills_file)
        
        # Extract walker number from HILLS.X filename
        match = re.search(r'HILLS\.(\d+)', filepath.name)
        if match:
            walker_num = int(match.group(1))
            walker_name = f"Walker {walker_num}"
        else:
            walker_name = filepath.name
        
        count = count_lines(filepath)
        line_counts[walker_name] = count
    
    return line_counts

def main():
    args = parse_args()
    
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
        #print(f"  {walker_name}: {count} lines ({percentage:.1f}%)")
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(args.csv, index=False)
    print(f"Mean completion: {df['percentage'].mean():.2f}%")
    print(f"\nSaved completion data to: {args.csv}")
    
    # Create bar plot
    sns.set_context("talk")
    fig, ax = plt.subplots(figsize=(max(10, len(data) * 0.8), 8))
    
    # Color bars based on completion percentage using Greens colormap
    cmap = plt.cm.YlGn
    # Normalize percentages to range 0-1 for colormap (use 0-100 range)
    norm = plt.Normalize(vmin=0, vmax=100)
    palette = [cmap(norm(pct)) for pct in df['percentage']]
    
    # Use seaborn barplot
    sns.barplot(
        data=df,
        x='walker',
        y='percentage',
        palette=palette,
        edgecolor='black',
        hue='walker',
        linewidth=1.5,
        ax=ax
    )
    
    # # Add value labels on bars
    # for i, (idx, row) in enumerate(df.iterrows()):
    #     ax.text(i, row['percentage'],
    #             f'{row["percentage"]:.1f}%\n({row["lines"]})',
    #             ha='center', va='bottom', fontsize=10)
    
    # Add reference line at 100%
    ax.axhline(y=100, color='red', linestyle='--', linewidth=2, alpha=0.7)#, label='100% (Expected)')
    
    # Formatting
    ax.set_xlabel('Walker')
    ax.set_ylabel('Completion (%)')
    # ax.set_title(f'Walker Completion Progress\n{folder_path.name}\n(Expected: {args.expected} lines)')
    ax.tick_params(axis='x', rotation=45)
    #ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Set y-axis to start at 0 and add some headroom
    max_pct = df['percentage'].max() if len(df) > 0 else 100
    ax.set_ylim(0, max(110, max_pct * 1.1))
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=DPI, bbox_inches='tight')
    print(f"Saved plot to: {args.output}")
    plt.close()

if __name__ == "__main__":
    main()
