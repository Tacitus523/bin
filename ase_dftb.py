#!/usr/bin/env python3
"""Run a DFTB+ calculation using the ASE DFTB calculator.

Usage:
    python ase_dftb.py structure.xyz [--system {dipeptide,thiol}] [--charge CHARGE]

Reads a structure file, runs DFTB3/3OB, and prints energy and forces.
Results are written to 'result.extxyz'.
"""

import argparse
from pathlib import Path
from typing import Dict

from ase import Atoms
from ase.calculators.dftb import Dftb
from ase.io import read, write


PARAM_DIR = "/lustre/home/ka/ka_ipc/ka_he8978/PARAMETERS"

SYSTEM_DEFAULTS: Dict[str, dict] = {
    "dipeptide": {
        "charge": 0,
        "prefix": f"{PARAM_DIR}/3ob-3-1/",
        "max_angular": {"C": '"p"', "N": '"p"', "O": '"p"', "H": '"s"'},
        "hubbard": {"C": -0.1492, "O": -0.1575, "N": -0.1535, "H": -0.1857},
        "temperature": 300.0,
    },
    "thiol": {
        "charge": -1,
        "prefix": f"{PARAM_DIR}/3ob-3-1_mod_S-S_B3LYP/",
        "max_angular": {"C": '"p"', "S": '"d"', "H": '"s"'},
        "hubbard": {"C": -0.1492, "S": -0.11, "H": -0.1857},
        "temperature": 500.0,
    },
}


def build_calculator(system: str, charge: int) -> Dftb:
    """Build ASE Dftb calculator for the given system type."""
    cfg = SYSTEM_DEFAULTS[system]
    if charge is None:
        charge = cfg["charge"]

    calc = Dftb(
        label="dftb_calc",
        Hamiltonian_SCC="Yes",
        Hamiltonian_Charge=charge,
        Hamiltonian_MaxSCCIterations=150,
        Hamiltonian_SCCTolerance=1e-5,
        Hamiltonian_ThirdOrderFull="Yes",
        Hamiltonian_HubbardDerivs_="",
        Hamiltonian_HCorrection_="Damping",
        Hamiltonian_HCorrection_Exponent=4.0,
        Hamiltonian_Filling_="Fermi",
        Hamiltonian_Filling_Temperature=cfg["temperature"] / 3.1668114e-6,  # K -> au
        Hamiltonian_MaxAngularMomentum_="",
        Hamiltonian_SlaterKosterFiles_="Type2FileNames",
        Hamiltonian_SlaterKosterFiles_Prefix=cfg["prefix"],
        Hamiltonian_SlaterKosterFiles_Separator='"-"',
        Hamiltonian_SlaterKosterFiles_Suffix='".skf"',
        **{
            f"Hamiltonian_MaxAngularMomentum_{el}": val
            for el, val in cfg["max_angular"].items()
        },
        **{
            f"Hamiltonian_HubbardDerivs_{el}": val
            for el, val in cfg["hubbard"].items()
        },
    )
    return calc


def run_dftb(atoms: Atoms, system: str, charge: int) -> Atoms:
    """Attach calculator, compute energy/forces, return atoms with results."""
    calc = build_calculator(system, charge)
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    print(f"Energy: {energy:.6f} eV")
    print(f"Max force component: {forces.max():.6f} eV/Ang")
    print(f"Number of atoms: {len(atoms)}")

    return atoms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DFTB+ via ASE")
    parser.add_argument("structure", type=str, help="Input structure file (.xyz, .extxyz, ...)")
    parser.add_argument(
        "--system",
        choices=list(SYSTEM_DEFAULTS.keys()),
        default="dipeptide",
        help="System preset (default: dipeptide)",
    )
    parser.add_argument("--charge", type=int, default=None, help="Override system charge")
    parser.add_argument(
        "--output", type=str, default="result.extxyz", help="Output file (default: result.extxyz)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    atoms = read(args.structure)
    atoms = run_dftb(atoms, args.system, args.charge)
    write(args.output, atoms)
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
