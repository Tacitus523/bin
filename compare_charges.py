#!/home/lpetersen/miniconda3/envs/kgcnn/bin/python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

LABELS = ["Mulliken", "Löwdin", "Hirshfeld", "ESP"]
ATOM_TYPES = ["C","S","H","H","H","C","S","H","H","H","C","S","H","H","H"]
# ATOM_TYPES = ["C",r"$C_O$","H","O",r"$C_N$","H","H","N",r"$C_N$","H","H","H","H","H","H"]

VAC_FILES = [
    "/data/lpetersen/thiol_disulfide/B3LYP_aug-cc-pVTZ_vacuum/scan/charges_mull.txt", 
    "/data/lpetersen/thiol_disulfide/B3LYP_aug-cc-pVTZ_vacuum/scan/charges_loew.txt", 
    "/data/lpetersen/thiol_disulfide/B3LYP_aug-cc-pVTZ_vacuum/scan/charges_hirsh.txt",
    "/data/lpetersen/thiol_disulfide/B3LYP_aug-cc-pVTZ_vacuum/scan/charges_esp.txt"      
]

WATER_FILES = [
    "/data/lpetersen/thiol_disulfide/B3LYP_aug-cc-pVTZ_water/scan/charges_mull.txt",     
    "/data/lpetersen/thiol_disulfide/B3LYP_aug-cc-pVTZ_water/scan/charges_loew.txt", 
    "/data/lpetersen/thiol_disulfide/B3LYP_aug-cc-pVTZ_water/scan/charges_hirsh.txt",
    "/data/lpetersen/thiol_disulfide/B3LYP_aug-cc-pVTZ_water/scan/charges_esp.txt"  
]

BINS = 30
ALPHA = 0.5

FIGSIZE = (21,9)
FONTSIZE = 24
LABELSIZE = 20
TITLE = False
DPI = 100

vacs = [np.loadtxt(file) for file in VAC_FILES]
waters = [np.loadtxt(file) for file in WATER_FILES]
labels = LABELS
atom_types = ATOM_TYPES

def plot_histogram(vacs, waters, labels, atom_types):
    print(vacs[0].shape)
    assert len(vacs)==len(waters)
    assert len(vacs)==len(labels)
    for data in vacs:
        assert data.shape == vacs[0].shape
    for data in waters:
        assert data.shape == vacs[0].shape

    diffs = [(waters[i] - vacs[i]) for i in range(len(vacs))]

    df = pd.DataFrame()
    df["Atom type"] = atom_types * len(vacs[0])

    type_labels = []
    for type_, type_suffix in zip([vacs, waters, diffs], [" vacuum", " water", " diff"]): 
        type_label = [label+type_suffix for label in labels]
        type_labels.append(type_label)
        for label, data in zip(type_label, type_):
            df[label] = data.flatten()
    vac_labels = type_labels[0]
    water_labels = type_labels[1]
    diff_labels = type_labels[2]
    df.sort_values("Atom type", inplace=True)

    means = df.groupby("Atom type").mean().reset_index().sort_values("Atom type")
    stds = df.groupby("Atom type").std().reset_index().sort_values("Atom type")

    for vac, water, diff, label in zip(vac_labels, water_labels, diff_labels, labels):
        fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, sharey=True)
        if TITLE:
            fig.suptitle(label + " Charge", fontsize=FONTSIZE)

        current_axis = axes[0]
        current_series = vac
        g = sns.histplot(data=df, x=current_series, hue="Atom type", bins=BINS, binrange=[-1.0, 0.5], alpha=ALPHA, ax=current_axis, stat="probability", common_norm=False)
        if TITLE:
            current_axis.set_title(r"$Q_{Vacuum}$", fontsize=LABELSIZE)
        current_axis.set_xlabel("Charge [e]", fontsize=LABELSIZE)
        current_axis.set_ylabel("Probability", fontsize=LABELSIZE)
        labels = [f"{means['Atom type'].iloc[i]: >2}: {means[current_series].iloc[i]:5.2f}\u00B1{stds[current_series].iloc[i]:.2f}" for i in range(len(means))]
        [text.set_text(label) for text, label in zip(g.axes.get_legend().texts, labels)]
        current_axis.tick_params(labelsize=LABELSIZE)
        plt.setp(current_axis.get_legend().get_texts(), fontsize=LABELSIZE)
        plt.setp(current_axis.get_legend().get_title(), fontsize=LABELSIZE)

        current_axis = axes[1]
        current_series = water
        g = sns.histplot(data=df, x=current_series, hue="Atom type", bins=BINS, binrange=[-1.0, 0.5], alpha=ALPHA, ax=current_axis, stat="probability", common_norm=False)
        if TITLE:
            current_axis.set_title(r"$Q_{Water}$", fontsize=LABELSIZE)
        current_axis.set_xlabel("Charge [e]", fontsize=LABELSIZE)
        labels = [f"{means['Atom type'].iloc[i]: >2}: {means[current_series].iloc[i]:5.2f}\u00B1{stds[current_series].iloc[i]:.2f}" for i in range(len(means))]
        [text.set_text(label) for text, label in zip(g.axes.get_legend().texts, labels)]
        current_axis.tick_params(labelsize=LABELSIZE)
        plt.setp(current_axis.get_legend().get_texts(), fontsize=LABELSIZE)
        plt.setp(current_axis.get_legend().get_title(), fontsize=LABELSIZE)

        current_axis = axes[2]
        current_series = diff
        g = sns.histplot(data=df, x=current_series, hue="Atom type", bins=BINS, binrange=[-0.25, 0.25], alpha=ALPHA, ax=current_axis, stat="probability", common_norm=False)
        if TITLE:
            current_axis.set_title(r"$Q_{Water} - Q_{Vacuum}$", fontsize=LABELSIZE)
        current_axis.set_xlabel(r"$\Delta$Charge [e]", fontsize=LABELSIZE)
        labels = [f"{means['Atom type'].iloc[i]: >2}: {means[current_series].iloc[i]:5.2f}\u00B1{stds[current_series].iloc[i]:.2f}" for i in range(len(means))]
        [text.set_text(label) for text, label in zip(g.axes.get_legend().texts, labels)]
        current_axis.tick_params(labelsize=LABELSIZE)
        plt.setp(current_axis.get_legend().get_texts(), fontsize=LABELSIZE)
        plt.setp(current_axis.get_legend().get_title(), fontsize=LABELSIZE)

        plt.savefig(label+"_charges_histogram", dpi=DPI)
        plt.close()

def plot_boxplot(vacs, waters, labels, atom_types):
    n_charge_types = len(vacs)
    n_molecules = len(vacs[0])
    n_datapoints = n_molecules*len(atom_types)
    env_labels_original = ["Vacuum", "Water", "Difference"]
    n_envs = len(env_labels_original)

    diffs = [(waters[i] - vacs[i]) for i in range(n_charge_types)]
    vacs = np.concatenate(vacs).flatten()
    waters = np.concatenate(waters).flatten()
    diffs = np.concatenate(diffs).flatten()
    charge_data = np.concatenate([vacs, waters, diffs])

    charge_labels = np.tile(np.repeat(labels, n_datapoints), n_envs)


    atom_type_labels = np.tile(np.array(atom_types * n_molecules), n_charge_types*n_envs)
    env_labels = np.repeat(env_labels_original, n_datapoints*n_charge_types)
    
    data = pd.DataFrame()
    data["Charge"] = charge_data
    data["Charge type"] = charge_labels
    data["Atom type"] = atom_type_labels
    data["Environment"] = env_labels
    for charge_type in labels:
        current_data = data[data["Charge type"]==charge_type]
        #current_data = current_data[data["Environment"].isin(env_labels_original[:2])]
        plt.figure(figsize=FIGSIZE)
        fig = sns.boxplot(data=current_data, x="Environment", y="Charge", hue="Atom type", order=env_labels_original)
        if TITLE:
            fig.axes.set_title("Charges boxplot", fontsize=FONTSIZE)
        fig.set_xlabel("Environment", fontsize=FONTSIZE)
        fig.set_ylabel("Charge", fontsize=FONTSIZE)
        fig.tick_params(labelsize=FONTSIZE, axis="x")
        fig.tick_params(labelsize=LABELSIZE, axis="y")
        plt.setp(fig.get_legend().get_title(), fontsize=FONTSIZE) # for legend title
        plt.setp(fig.get_legend().get_texts(), fontsize=FONTSIZE) # for legend text
        plt.tight_layout()
        plt.savefig(charge_type+"_charges_boxplot.png", dpi=DPI)

if __name__=="__main__":
    plot_histogram(vacs, waters, labels, atom_types)
    plot_boxplot(vacs, waters, labels, atom_types)
