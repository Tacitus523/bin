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

H_to_kcal_mol = 627.509

def main():
    ap = argparse.ArgumentParser(description="Quickly plot a 1D plot")
    ap.add_argument("-f", type=str, dest="file", action="store", required=True, help="File with data to plot", metavar="file")
    ap.add_argument("-s", type=str, dest="data_source_file", action="store", required=False, default=None, help="File with data sources for coloring", metavar="data source file")
    args = ap.parse_args()
    file = args.file
    data_source_file = args.data_source_file

    data = pd.DataFrame()
    data["Energy"] = np.loadtxt(file)*H_to_kcal_mol
    if data_source_file is not None:
        data["Data Source"] = pd.read_csv(data_source_file)
    else:
        data["Data Source"] = ["Unknown Source"]*len(data)
    data["Data Source"] = data["Data Source"].astype("category")
    data["Index"] = data.index

    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot()
    sns.scatterplot(data=data, x="Index", y="Energy", hue="Data Source", marker=".")
    ax.set_xlabel("Data point", fontsize=FONTSIZE)
    ax.set_ylabel(r"Energy [$\frac{\text{kcal}}{\text{mol}}$]", fontsize=FONTSIZE)
    plt.tick_params(axis='both', which="major", labelsize=LABELSIZE)
    legend = plt.legend(fontsize=FONTSIZE, title_fontsize=FONTSIZE, bbox_to_anchor=(1.05, 1))
    for legend_handle in legend.legend_handles: 
        legend_handle.set_alpha(1)
        legend_handle.set_markersize(MARKERSIZE)
    plt.tight_layout()
    plt.savefig("data_points_vs_energy.png", dpi=DPI)
    plt.close()

    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot()

    sns.boxplot(data=data, x="Data Source", y="Energy")
    ax.set_xlabel("Data Source", fontsize=FONTSIZE)
    ax.set_ylabel(r"Energy [$\frac{\text{kcal}}{\text{mol}}$]", fontsize=FONTSIZE)
    #ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
    ax.tick_params(axis='x', labelrotation=45)
    plt.tick_params(axis='both', which="major", labelsize=LABELSIZE)
    plt.tight_layout()
    plt.savefig("data_points_vs_energy_boxplot.png", dpi=DPI)
    plt.close()

main()