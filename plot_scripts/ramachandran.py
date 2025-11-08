#!/home/lpetersen/miniconda3/envs/mda/bin/python
import argparse

import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Dihedral, Ramachandran
import numpy as np
import pandas as pd
import seaborn as sns

# Reads a trajectroy using MdAnalysis, calculates and saves the phi and psi angles for a ramachandran plot

TRAJECTORY_FILE_NAME = "geoms.xyz"
SAVE_FILE_NAME = "ramachandran.dat"

DPI = 300

def read_trajectory(trajectory_file_name: str):
    """
    Read trajectory from a file
    """
    u = mda.Universe(trajectory_file_name)
    return u


def do_ramachandran_analysis(SAVE_FILE_NAME, universe: mda.Universe):
    atoms = universe.select_atoms("backbone") # Only works with .ttr/.xtc files, which have resname info
    ram = Ramachandran(atoms)
    ram.run()
    ram.save(SAVE_FILE_NAME)

def do_dihedral_analysis(SAVE_FILE_NAME, universe: mda.Universe, ):
    atoms = [universe.atoms[[4, 6, 8, 14]], universe.atoms[[6, 8, 14, 16]]]
    dih = Dihedral(atoms)
    dih.run()
    np.savetxt(SAVE_FILE_NAME, dih.results.angles, fmt="%.1f")
    return dih

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--traj", dest="traj", required=False, help="Path to trajectory file, .xyz format expected", metavar="Trajectory file")
    args = parser.parse_args()

    if args.traj:
        trajectory_file_name = args.traj
    else:
        trajectory_file_name = TRAJECTORY_FILE_NAME

    universe: mda.Universe = read_trajectory(trajectory_file_name)
    #do_ramachandran_analysis(SAVE_FILE_NAME, u)
    dih: Dihedral = do_dihedral_analysis(SAVE_FILE_NAME, universe)
    phi = dih.results.angles[:,0]
    psi = dih.results.angles[:,1]

    phi = np.where(phi<0, phi+360, phi)
    psi = np.where(psi<0, psi+360, psi)

    # energy_minimum_names = [r"$C_5^{ext}$", r"$C_7^{eq}$", r"$\beta_2$", r"$\alpha$", r"$\alpha'$", r"$\alpha_L$", r"$C^{ax}_7$"]
    # # [(phi_min,phi_max),(psi_min,psi_max)] per minimum
    # dihedral_ranges = [[(180, 220),(160,200)],[(240,320),(30,100)],[(220,240),(10,40)],[(290,310),(320,350)],[(170,220),(260,320)],[(50,80),(10, 50)],[(50,100),(270,320)]]
    energy_minimum_names = [r"$C_5^{ext}$", r"$C_7^{eq}$", r"$\alpha'$", r"$C^{ax}_7$"]
    # [(phi_min,phi_max),(psi_min,psi_max)] per minimum
    dihedral_ranges = [[(180, 220),(160,200)],[(240,320),(30,100)],[(170,220),(260,320)],[(50,100),(270,320)]]

    sns.set_context("talk")
    plt.scatter(phi, psi)
    plt.xlabel(r"$\varphi$ [$^\circ$]")
    plt.ylabel(r"$\psi$ [$^\circ$]")
    for energy_minumum_name, dihedral_range in zip(energy_minimum_names, dihedral_ranges):
        plt.text(np.mean(dihedral_range[0]), np.mean(dihedral_range[1]), energy_minumum_name, fontsize=12)
    plt.tight_layout()
    plt.savefig("dihedral_scatter.png", dpi=DPI)
    plt.close()

    plt.hist2d(phi, psi, bins=50)
    plt.xlabel(r"$\varphi$ [$^\circ$]")
    plt.ylabel(r"$\psi$ [$^\circ$]")
    for energy_minumum_name, dihedral_range in zip(energy_minimum_names, dihedral_ranges):
        plt.text(np.mean(dihedral_range[0]), np.mean(dihedral_range[1]), energy_minumum_name, fontsize=12)
    plt.tight_layout()
    plt.savefig("dihedral_hist2d.png", dpi=DPI)
    plt.close()


    energy_minimum_labels = np.array(["undefined"]*len(phi))
    for energy_minimum_name, (phi_range, psi_range) in zip(energy_minimum_names, dihedral_ranges):
        energy_minimum_labels = np.where((phi>phi_range[0]) & (phi<phi_range[1]) & (psi>psi_range[0]) & (psi<psi_range[1]), energy_minimum_name, energy_minimum_labels)
    np.savetxt("dihedral_labels.dat", energy_minimum_labels, fmt="%s")

    df = pd.DataFrame({
        r"$\varphi$ [$^\circ$]": phi,
        r"$\psi$ [$^\circ$]": psi,
        "Energy Minimum": energy_minimum_labels
    })

    figure = plt.figure(figsize=(12,8))
    ax = sns.scatterplot(data=df, x=r"$\varphi$ [$^\circ$]", y=r"$\psi$ [$^\circ$]", hue="Energy Minimum", alpha=0.5, marker=".")
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig("dihedral_scatter_labeled.png", dpi=DPI)
    plt.close()

    fig = plt.figure(figsize=(16,9))
    ax = sns.histplot(data=df, x=r"$\varphi$ [$^\circ$]", y=r"$\psi$ [$^\circ$]", hue="Energy Minimum", bins=150, alpha=0.5)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig("dihedral_hist2d_labeled.png",dpi=DPI)
    plt.close()