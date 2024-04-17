#!/lustre/home/ka/ka_ipc/ka_he8978/miniconda3/envs/kgcnn_new/bin/python3
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

FONTSIZE = 30
LABELSIZE = 18
MARKERSIZE = 18
DPI=400

def main():
    ap = argparse.ArgumentParser(description="Plot a scatterplot and a boxplot")
    ap.add_argument("-f", type=str, dest="file", action="store", required=True, help="File with data to plot", metavar="file")
    args = ap.parse_args()
    file = args.file

    data = pd.read_csv(file, sep=";")

    percentile = data["Timestep"].quantile(0.75)
    max_value = data["Timestep"].max()
    need_to_zoom = (10*percentile < max_value)

    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot()

    sns.boxplot(data=data, x="Iteration", y="Timestep")
    ax.set_xlabel("Iteration", fontsize=FONTSIZE)
    ax.set_ylabel("Timesteps", fontsize=FONTSIZE)
    ax.tick_params(axis='x', labelrotation=45)
    plt.tick_params(axis='both', which="major", labelsize=LABELSIZE)
    plt.tight_layout()
    plt.savefig("iteration_vs_timestep_boxplot.png", dpi=DPI)
    if need_to_zoom:
        ax.set_ylim(0, percentile*10)
        plt.savefig("iteration_vs_timestep_zoomed_boxplot.png", dpi=DPI)

    plt.close()

main()