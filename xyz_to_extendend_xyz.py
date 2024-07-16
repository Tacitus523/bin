import os
import numpy as np

DATA_FOLDER: str = "/lustre/work/ws/ws1/ka_he8978-dipeptide/training_data/B3LYP_aug-cc-pVTZ_vacuum"
GEOM_FILE: str = "geoms.xyz" # Angstrom units to Angstrom units
ENERGY_FILE: str = "energies.txt" # Hartree to eV
FORCE_FILE: str = "forces_conv.xyz" # Hartree/Bohr to eV/Angstrom, xyz format, assumed to be not actually forces but gradients, transformed to forces by multiplying by -1
CHARGE_FILE: str = "charges.txt" # e to e
OUTFILE: str = "geoms.extxyz"

TOTAL_CHARGE: float = 0.0 # e to e
BOXSIZE: float = 3.0 # nm to Angstrom, assuming cubic box, irrelevant unless periodic system

H_to_eV = 27.211386245988
H_B_to_eV_A = 51.422086190832
e_to_e = 1.0
nm_to_A = 10.0

def read_xyz(filename):
    with open(filename, 'r') as file:
        while True:
            try:
                n_atoms = int(file.readline())
                comment = file.readline()
                atoms = [file.readline().strip().split() for _ in range(n_atoms)]
                yield n_atoms, comment, atoms
            except ValueError:  # End of file
                break

def load_energy_charge_data(energy_file, charge_file):
    energies = np.loadtxt(energy_file)
    charges = np.loadtxt(charge_file)
    return energies, charges

def load_force_data(force_file):
    forces = []
    with open(force_file, 'r') as file:
        while True:
            try:
                n_atoms = int(file.readline())
                file.readline()  # Skip comment
                forces.append([list(map(float, file.readline().strip().split()[-3:])) for _ in range(n_atoms)])
            except ValueError:  # End of file
                break
    return forces

def write_extxyz(outfile, molecules, energies, forces, charges):
    boxsize = BOXSIZE*nm_to_A 
    lattice_vector = f'{boxsize:0.1f} 0.0 0.0 0.0 {boxsize:0.1f} 0.0 0.0 0.0 {boxsize:0.1f}'
    with open(outfile, 'w') as file:
        for mol_idx, (n_atoms, comment, atoms) in enumerate(molecules):
            file.write(f"{n_atoms}\n")
            file.write(f'Lattice="{lattice_vector}" Properties=species:S:1:pos:R:3:charge:R:1:force:R:3 energy={energies[mol_idx]} total_charge={TOTAL_CHARGE} pbc="F F F" comment="{comment}"')
            for at_idx, atom in enumerate(atoms):
                atom_line = " ".join(atom[:4])  # Assuming atom format is [element, x, y, z]
                force_line = " ".join(map(lambda x: f"{x: .8f}", forces[mol_idx][at_idx]))
                charge = charges[mol_idx][at_idx]
                file.write(f"{atom_line} {charge} {force_line}\n")

def main():
    geom_file = os.path.join(DATA_FOLDER, GEOM_FILE)
    energy_file = os.path.join(DATA_FOLDER, ENERGY_FILE)
    force_file = os.path.join(DATA_FOLDER, FORCE_FILE)
    charge_file = os.path.join(DATA_FOLDER, CHARGE_FILE)
    outfile = os.path.join(DATA_FOLDER, OUTFILE)

    molecules = list(read_xyz(geom_file))
    energies, charges = load_energy_charge_data(energy_file, charge_file)
    forces: list = load_force_data(force_file) # Possibly ragged list
    energies *= H_to_eV
    charges *= e_to_e
    forces = [[[force * H_B_to_eV_A * -1 for force in atom] for atom in molecule] for molecule in forces]
    write_extxyz(outfile, molecules, energies, forces, charges)

if __name__ == "__main__":
    main()

    