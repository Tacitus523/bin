#!/home/lpetersen/miniconda3/envs/mda/bin/python3
import warnings
warnings.filterwarnings('ignore')

import argparse
import os
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import distances
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Script to plot the correlation plots E_elec calculations from point charges and the true E_elec of QM energies in QM/MM calculations.The vacuum energies of the exact same structures are also requiered to 
# Also able to compare energies without any additional calculations

ALPHA=0.1
FONTSIZE = 30
LABELSIZE = 18
MARKERSIZE = 18
DPI=400
DEFAULT_NAME = "E_elec_correlation.png"

volt_to_atomic_units = 1/27.211386245988
H_to_kcal_mol = 627.509

def main():
    ap = argparse.ArgumentParser(description="Energy correlation between vacuum and water(or simply two energy files)")
    ap.add_argument("-v", type=str, dest="vacuum_energy_file", action="store", required=True, help="File with vacuum energy data", metavar="vacuum energy file")
    ap.add_argument("-e", type=str, dest="env_energy_file", action="store", required=False, default=None, help="File with energy data in environment [H]", metavar="environment energy file") # in Hartree
    ap.add_argument("-c", type=str, dest="env_charge_file", action="store", required=False, default=None, help="File with charges in environment [e], default: None", metavar="env charge file") # in Hartree
    ap.add_argument("-p", type=str, dest="env_esp_file", action="store", required=False, default=None, help="File with electrostatic potentials from environment [V], default: None", metavar="ESP file") # in Volt
    ap.add_argument("-s", type=str, dest="data_source_file", action="store", required=False, default=None, help="File with data sources for coloring", metavar="data source file, default: ''")
    ap.add_argument("-t", type=str, dest="title", action="store", required=False, default=None, help="Title of the plot", metavar=f"title, default: None")
    ap.add_argument("-o", type=str, dest="out_file", action="store", required=False, default=None, help="File name of the saved figure", metavar=f"out file, default: {DEFAULT_NAME}")
    args = ap.parse_args()

    vacuum_energy_file = args.vacuum_energy_file
    vacuum_energies = np.loadtxt(vacuum_energy_file)

    if args.env_energy_file is not None:
        env_energy_file = args.env_energy_file
        env_energies = np.loadtxt(env_energy_file)
        assert env_energies.shape == vacuum_energies.shape, "Amount of energy entries do not match."
    else:
        env_energy_file = None
        env_energies = np.zeros_like(vacuum_energies)

    if args.env_charge_file is not None:
        env_charge_file = args.env_charge_file
        env_charges = np.loadtxt(env_charge_file)
        assert len(vacuum_energies) == len(env_charges), "Amount of energy entries do not match."
    else:
        env_charge_file = None
        env_charges = np.zeros_like(vacuum_energies[:,np.newaxis])

    if args.env_esp_file is not None:
        env_esp_file = args.env_esp_file
        env_esps = np.loadtxt(env_esp_file)*volt_to_atomic_units
        assert env_charges.shape == env_esps.shape, "Charge and ESP shapes do not match."
    else:
        env_esp_file = None
        env_esps = np.zeros_like(vacuum_energies[:,np.newaxis])

    assert (env_charge_file is not None and env_esp_file is not None) or (env_charge_file is None and env_esp_file is None), "Only got one out of Charge and ESP-File. Should be both or neither."

    if args.data_source_file is not None:
        data_source_file = args.data_source_file
        data = pd.read_csv(data_source_file, names=["Data Source"])
    else:
        data = pd.DataFrame()

    title = args.title
    if args.out_file is not None:
        out_file = args.out_file
    else:
        out_file = DEFAULT_NAME

    true_e_elec = -(env_energies - vacuum_energies) 
    calc_e_elec = np.sum(env_charges*env_esps, axis=-1) 

    if env_charge_file is not None:
        energy_1 = true_e_elec *H_to_kcal_mol
        energy_2 = calc_e_elec *H_to_kcal_mol
    else:
        energy_1 = vacuum_energies *H_to_kcal_mol
        energy_2 = env_energies *H_to_kcal_mol

    mean_error = np.mean(energy_2 - energy_1)
    mae = mean_absolute_error(energy_1, energy_2)
    rmse = mean_squared_error(energy_1, energy_2, squared=False)
    r2 = r2_score(energy_1, energy_2)

    data["True E_elec"] = energy_1 
    data["Calculated E_elec"] = energy_2

    fig = plt.figure(figsize=(16,16))
    ax = fig.add_subplot()
    if args.data_source_file is not None:
        plot = sns.scatterplot(x="True E_elec", y="Calculated E_elec", data=data,
            hue="Data Source", palette=sns.color_palette("hls", 10),
            legend="full", alpha=0.1)

        legend = plt.legend(title='Data Source', loc='upper left', fontsize=FONTSIZE, title_fontsize=FONTSIZE)
        for legend_handle in legend.legendHandles: 
            legend_handle.set_alpha(1)
            legend_handle.set_markersize(MARKERSIZE)
    else:
        plot = sns.scatterplot(x="True E_elec", y="Calculated E_elec", data=data,
            palette=sns.color_palette("hls", 10), alpha=ALPHA)

    value_min = data[["True E_elec","Calculated E_elec"]].min().min()
    value_max = data[["True E_elec","Calculated E_elec"]].max().max()
    plt.plot([value_min-1, value_max+1], [value_min-1, value_max+1], "k")
    plot.set(xlim=(value_min, value_max))
    plot.set(ylim=(value_min, value_max))

    text_x = 0.75*(value_max-value_min) + value_min
    text_y = 0.1*(value_max-value_min) + value_min
    ax.text(text_x, text_y, f"MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nR2: {r2:.2f}", fontdict={"fontsize": LABELSIZE}, bbox={
        "facecolor": "grey", "alpha": 0.5, "pad": 10})


    plt.title(title, {'fontsize': FONTSIZE})
    ax.set_xlabel(r'True $\text{E}_\text{elec}$ [$\frac{\text{kcal}}{\text{mol}}$]', fontsize=FONTSIZE)
    ax.set_ylabel(r'Calculated $\text{E}_\text{elec}$ [$\frac{\text{kcal}}{\text{mol}}$]', fontsize=FONTSIZE)
    plt.tick_params(axis='both', which="major", labelsize=LABELSIZE)

    plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    plt.tight_layout()
    plt.savefig(out_file, dpi=DPI, bbox_inches = "tight")

main()