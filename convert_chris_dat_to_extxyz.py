#!/usr/bin/env python3
import argparse
import os
from typing import List, Tuple

import ase
import ase.io
"""
Format .dat:
1st line per configuration: electronic energy, repulsive energy, excitation energy; units: Hartree, Hartree, Hartree
Following lines: atomic number, x, y, z, ESP, fx, fy, fz; units: None, Angstrom, Angstrom, Angstrom, Hartree/e, Hartree/Bohr, Hartree/Bohr, Hartree/Bohr

Format .extxyz:
1st line: number of atoms
2nd line: Information about the molecular properties with respective keys; total energy: energy, electronic energy: electronic_energy, repulsive energy: repulsive_energy, excitation energy: excitation_energy;
units: Hartree, Hartree, Hartree, Hartree
Following lines: atomic symbol, x, y, z, fx, fy, fz, ESP; units: None, Angstrom, Angstrom, Angstrom, eV/Angstrom, eV/Angstrom, eV/Angstrom, eV/e
"""

DEFAULT_OUTPUT_FILE = "{}.extxyz"

# Conversion factors
H_to_eV = 27.211386245988
H_B_to_eV_A = 51.422086190832
nm_to_A = 10

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Converts a .dat file from the custom format to an extended xyz file")
    parser.add_argument("-i","--input_file", required=True, type=str, help="Input .dat file")
    parser.add_argument("-o", "--output_file", required=False, default=None, type=str, help="Output .extxyz file")
    parser.add_argument("-b", "--box_size", required=False, default=None, type=float, help="Box size in nm for periodic boundary conditions, assuming cubic box")
    args = parser.parse_args()
    if args.output_file is None:
        input_file_basename = os.path.splitext(os.path.basename(args.input_file))[0]
        args.output_file = DEFAULT_OUTPUT_FILE.format(input_file_basename)
    return args

def read_dat_file(args: argparse.Namespace) -> Tuple[List[Tuple[str, float, float, float, float, float, float, float]], List[List[float]]]:
    with open(args.input_file, "r") as f:
        lines = f.readlines()

    atomic_symbols = ase.data.chemical_symbols


    configs = []  # Speichert alle Konfigurationen
    energies = []  # Speichert die Energieinformationen als 2er/3er-Tupel(electronic energy, repulsive energy(, excitation energy))
    lin_idx = 0
    mol_idx = 0
    while lin_idx < len(lines):
        current_config = [] # Temporärer Speicher für die aktuelle Konfiguration
        current_energies = []  # Temporärer Speicher für die aktuelle Energieinformation

        line = lines[lin_idx]
        values = line.split()
        if not values:
            lin_idx += 1
            continue
        current_energies = map(float, values)
        energies.append(current_energies)
        lin_idx += 1
        
        line = lines[lin_idx]
        while line.strip() != "":
            values = line.split()
            atomic_num = int(values[0])
            symbol = atomic_symbols[atomic_num]
            x, y, z = map(float, values[1:4])  # Koordinaten
            esp = float(values[4])  # ESP
            fx, fy, fz = map(float, values[5:8])  # Kräfte
            current_config.append((symbol, x, y, z, esp, fx, fy, fz))
            lin_idx += 1
            if lin_idx < len(lines):
                line = lines[lin_idx]
            else:
                break
        mol_idx += 1
        if mol_idx % 1000 == 0:
            print(f"Reading molecule {mol_idx}")
        configs.append(current_config)
    return configs, energies

def convert_configuration_units(configs: List[Tuple[str, float, float, float, float, float, float, float]]) -> List[Tuple[str, float, float, float, float, float, float, float]]:
    """Converts the units of a configuration from the custom format to the extended xyz format"""
    def do_conversion(config):
        converted_config = []
        for config_line in config:
            symbol, x, y, z, esp, fx, fy, fz = config_line
            esp *= H_to_eV
            fx *= H_B_to_eV_A
            fy *= H_B_to_eV_A
            fz *= H_B_to_eV_A
            converted_config.append((symbol, x, y, z, esp, fx, fy, fz))
        return converted_config
    
    return [do_conversion(config) for config in configs]

def convert_energies_units(energies: List[List[float]]) -> List[List[float]]:
    """Converts the units of the energies from the custom format to the extended xyz format"""
    return [[energy * H_to_eV for energy in energy_list] for energy_list in energies]
    
def write_extxyz_file(args: argparse.Namespace, configs: List[Tuple[str, float, float, float, float, float, float, float]], energies: List[List[float]]) -> None:
    output_file_name = args.output_file
    output_file = open(output_file_name, "w")
    mol_idx = 0
    for energy, config in zip(energies, configs):
        num_atoms = len(config)
        output_file.write(f"{num_atoms}\n")
        electronic_energy, repulsive_energy = energy[0], energy[1]
        try:
            excitation_energy = energy[2]
        except IndexError:
            excitation_energy = None
        state_energy = electronic_energy + repulsive_energy

        if excitation_energy is not None:
            state_energy += excitation_energy

        info_line = f"Properties=species:S:1:pos:R:3:ref_force:R:3:esp:R:1 ref_energy={state_energy} electronic_energy={electronic_energy} repulsive_energy={repulsive_energy}"
        if excitation_energy is not None:
            info_line += f" excitation_energy={excitation_energy}"
        if args.box_size is not None:
            info_line += f" Lattice=\"{args.box_size*nm_to_A} 0.0 0.0 0.0 {args.box_size*nm_to_A} 0.0 0.0 0.0 {args.box_size*nm_to_A}\""
        info_line += " pbc=\"F F F\"\n"
        output_file.write(info_line)
        
        for atom in config:
            symbol, x, y, z, esp, fx, fy, fz = atom
            output_file.write(f"{symbol:<2} {x:.6f} {y:.6f} {z:.6f} {fx:+.9f} {fy:+.9f} {fz:+.9f} {esp:+.9f}\n")
        
        mol_idx += 1
        if mol_idx % 1000 == 0:
            print(f"Writing molecule {mol_idx}")
    output_file.close()


    try:
        universe = ase.io.read(output_file_name, index=0, format="extxyz") # Check if the file can be read by reading the first entry
    except Exception as e:
        print(f"Error reading .extxyz: {e}")
    else:
        print(f"Saved .extxyz: {output_file_name}")

def main():
    args = parse_args()
    configs, energies = read_dat_file(args)
    configs = convert_configuration_units(configs)
    energies = convert_energies_units(energies)
    write_extxyz_file(args, configs, energies)


if __name__ == "__main__":
    main()




