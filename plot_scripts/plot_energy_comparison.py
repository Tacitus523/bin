#!/home/lpetersen/miniconda3/envs/kgcnn/bin/python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

ENERGY_VACUUM_FILE = "/data/lpetersen/thiol_disulfide/B3LYP_aug-cc-pVTZ_vacuum/redo_exact_strucutres/energies.txt" # in Hartree
ENERGY_WATER_FILE = "/data/lpetersen/thiol_disulfide/B3LYP_aug-cc-pVTZ_water/scan_with_gbw/e_water_e_elec_difference.txt" # in Hartree

BINS = 40
ALPHA = 0.5

FIGSIZE = (16,9)
FONTSIZE = 20
LABELSIZE = 16
TITLE = False
DPI = 600

H_to_kcal_mol = 627.509

energy_vacuum = np.loadtxt(ENERGY_VACUUM_FILE)*H_to_kcal_mol
energy_water = np.loadtxt(ENERGY_WATER_FILE)*H_to_kcal_mol
energy_diff = energy_water - energy_vacuum
energies = [energy_vacuum, energy_water, energy_diff]
labels = [r"$E_{Vacuum}$", r"$E_{Water}$", r"\Delta E"]


df = pd.DataFrame()
for energy, label in zip(energies, labels):
    df[label] = energy

plt.figure(figsize=FIGSIZE)
fig = sns.histplot(data=df[[labels[0],labels[1]]], bins=BINS, alpha=ALPHA)
if TITLE:
    fig.axes.set_title("Energy distribution", fontsize=FONTSIZE)
fig.set_xlabel(r"Energy [$\frac{kcal}{mol}$]", fontsize=FONTSIZE)
fig.set_ylabel("Count", fontsize=FONTSIZE)
fig.tick_params(labelsize=LABELSIZE)
plt.setp(fig.get_legend().get_texts(), fontsize=LABELSIZE)
plt.setp(fig.get_legend().get_title(), fontsize=LABELSIZE)
fig.xaxis.get_offset_text().set_fontsize(LABELSIZE)
plt.tight_layout()
plt.savefig(f"energy_histogram.png", dpi=DPI)
plt.close()

plt.figure(figsize=FIGSIZE)
fig = sns.histplot(data=energy_diff, bins=BINS, alpha=ALPHA)
if TITLE:
    fig.axes.set_title("Energy difference distribution", fontsize=FONTSIZE)
fig.set_xlabel(r"Energy difference [$\frac{kcal}{mol}$]", fontsize=FONTSIZE)
fig.set_ylabel("Count", fontsize=FONTSIZE)
fig.tick_params(labelsize=LABELSIZE)
fig.xaxis.get_offset_text().set_fontsize(LABELSIZE)
plt.tight_layout()
plt.savefig(f"energy_histogram.png", dpi=DPI)
plt.close()

n_datapoints= len(df)
labels = [[label]*n_datapoints for label in labels[:2]]
energies = np.concatenate(energies[:2])
labels = np.concatenate(labels)
data = pd.DataFrame()
data["Energy"] = energies
data["Labels"] = labels
    
plt.figure(figsize=FIGSIZE)
fig = sns.boxplot(data=data, x="Labels", y="Energy", hue=None)
if TITLE:
    fig.axes.set_title("Energy boxplot", fontsize=FONTSIZE)
fig.set_xlabel("Energy source", fontsize=FONTSIZE)
fig.set_ylabel(r"Energy [$\frac{kcal}{mol}$]", fontsize=FONTSIZE)
fig.tick_params(labelsize=FONTSIZE, axis="x")
fig.tick_params(labelsize=LABELSIZE, axis="y")
fig.yaxis.get_offset_text().set_fontsize(LABELSIZE)
plt.tight_layout()
plt.savefig(f"energy_boxplot.png", dpi=DPI)
plt.close()

plt.figure(figsize=FIGSIZE)
fig = sns.violinplot(data=data, x="Labels", y="Energy", hue=None, fontsize=FONTSIZE)
if TITLE:
    fig.axes.set_title("Energy violinplot", fontsize=FONTSIZE)
fig.set_xlabel("Energy source", fontsize=FONTSIZE)
fig.set_ylabel(r"Energy [$\frac{kcal}{mol}$]", fontsize=FONTSIZE)
fig.tick_params(labelsize=FONTSIZE, axis="x")
fig.tick_params(labelsize=LABELSIZE, axis="y")
fig.yaxis.get_offset_text().set_fontsize(LABELSIZE)
plt.tight_layout()
plt.savefig(f"energy_violinplot.png", dpi=DPI)
plt.close()
