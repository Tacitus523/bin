#!/lustre/home/ka/ka_ipc/ka_he8978/miniconda3/envs/kgcnn_new/bin/python3
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json

FONTSIZE = 30
LABELSIZE = 18
MARKERSIZE = 18
FIGSIZE=(20,10)
DPI=400

def main():
    ap = argparse.ArgumentParser(description="Plot a scatterplot and a boxplot")
    ap.add_argument("-f", type=str, dest="file_name", action="store", required=True, help="File with data to plot", metavar="file_name")
    args = ap.parse_args()
    file_name = args.file_name

    with open(file_name, "r") as f:
        error_dict_list: list[dict] = json.load(f)

    error_dict_flat = {}
    for error_dict in error_dict_list:
        for key, value in error_dict.items():
            if key in error_dict_flat:
                error_dict_flat[key].append(value)
            else:
                error_dict_flat[key] = [value]
    for key, value in error_dict_flat.items():
        error_dict_flat[key] = np.array(value, dtype=np.float32)

    error_df = pd.DataFrame(error_dict_flat)

    num_cols = len(error_df.columns)
    num_cols_per_row = 2
    num_rows = (num_cols + 1) // num_cols_per_row

    sns.set_context(context="talk", font_scale=1.3)
    fig, axes = plt.subplots(num_rows, num_cols_per_row, figsize=FIGSIZE)
    if num_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, col in enumerate(error_df.columns):
        row_idx = idx // num_cols_per_row
        col_idx = idx % num_cols_per_row
        ax = axes[row_idx, col_idx]
        error_df[col].plot(ax=ax, title=error_df[col].name)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(error_df[col].name)

    plt.tight_layout()
    plt.savefig("iteration_vs_error.png", dpi=DPI)
    plt.close()

if __name__=="__main__":
    main()