#!/usr/bin/env python3
import os
import argparse

import numpy as np
from scipy.spatial.distance import cdist

angstrom_to_bohr = 1.8897259886
atomic_units_to_volt = 27.211386245988

# for testing
testing = False

def calculate_esp_and_esp_gradient(qm_coords, mm_coords, mm_charges):
    qmmm_distances = cdist(qm_coords, mm_coords) # shape: (n_qm_atoms, n_mm_atoms)
    esps = np.matmul(1/qmmm_distances, mm_charges) # shape: (n_qm_atoms, 1)
    esps = esps.reshape((1,-1)) # shape: (1, n_qm_atoms)
    
    directions = (qm_coords[:, np.newaxis, :] - mm_coords[np.newaxis, :, :]) / qmmm_distances[:, :, np.newaxis] # shape: (n_qm_atoms, n_mm_atoms, 3)
    gradient_magnitudes = mm_charges[np.newaxis, :] / qmmm_distances**2 # shape: (n_qm_atoms, n_mm_atoms)
    gradients = -1*np.sum(directions * gradient_magnitudes[:, :, np.newaxis], axis=1) # shape: (n_qm_atoms, 3)
    return esps, gradients
    

def main():
    ap = argparse.ArgumentParser(description="Calculates the ESP from MM-atoms on QM-atoms from Orca-input and -pointcharge files")
    ap.add_argument("-d", "--dir", default=None, type=str, dest="folder_prefix", action="store", required=False, help="Prefix of the directionaries with the orca-calculations, default: None", metavar="folder_prefix")
    ap.add_argument("-i", "--input", type=str, dest="input_prefix", action="store", required=True, help="Prefix of the Input-file for the orca-calculation", metavar="file_prefix")
    ap.add_argument("-u", "--unit", choices=["V", "au"], default="au", type=str, dest="unit", action="store", required=False, help="Unit of the ESP, default: atomic units(au)", metavar="unit")
    args = ap.parse_args()
    folder_prefix = args.folder_prefix
    input_prefix = args.input_prefix
    unit= args.unit

    input_file = input_prefix + ".inp"
    point_charge_file = input_prefix + ".pc"
    orcainfo_file = input_prefix + ".ORCAINFO"

    if folder_prefix is not None:
        sp_calculation_location = os.path.dirname(folder_prefix)
        sp_calculation_prefix = os.path.basename(folder_prefix)
        if not sp_calculation_location:
            sp_calculation_location = "."
        folders = [os.path.join(sp_calculation_location, f.name) for f in os.scandir(sp_calculation_location) if f.is_dir() and f.name.startswith(sp_calculation_prefix)]
    else:
        folders = [os.getcwd()]
    original_folder = os.path.abspath(os.getcwd())
    for folder in folders:
        os.chdir(folder)
        
        with open(orcainfo_file, "r") as f:
            num_header_lines = sum(1 for line in f) + 3

        qm_coords = np.genfromtxt(input_file, skip_header=num_header_lines, skip_footer=2, usecols=(1,2,3))
        mm_coords = np.genfromtxt(point_charge_file, skip_header=1, usecols=(1,2,3))
        mm_charges = np.genfromtxt(point_charge_file, skip_header=1, usecols=(0,))

        esps, gradients = calculate_esp_and_esp_gradient(qm_coords, mm_coords, mm_charges)

        if unit=="au":
            esps = esps/angstrom_to_bohr # from e/Angstrom to e/bohr(atomic unit)
        elif unit=="V":
            esps = esps/angstrom_to_bohr*atomic_units_to_volt # from e/Angstrom to e/Bohr(atomic unit) to V
            
        gradients = gradients/angstrom_to_bohr/angstrom_to_bohr # from e/Angstrom^2 to e/Bohr^2 

        np.savetxt("esps_by_mm.txt", esps, fmt='%3.5f')
        np.savetxt("esp_gradients.txt", gradients, fmt='%3.7f')
        os.chdir(original_folder)

def test_function():
    qm_coords = np.array([
        [109.3700000,  109.9500000,  111.0900000],
        [110.4900000,  110.1700000,  109.6400000],
    ])
    mm_coords = np.array([
        [101.6425487,  102.9492463,  105.4171885],
        [102.2191410,  103.4443143,  105.9991515],
        [102.2375458,  102.4525691,  104.8554750]
    ])
    mm_charges = np.array([
        -0.8340,
        0.4170,
        0.4170
    ])
    
    test_esps, test_gradients = calculate_esp_and_esp_gradient(qm_coords, mm_coords, mm_charges)
    
    target_esps = np.array([[0.00242374, 0.00250612]])
    assert np.allclose(test_esps, target_esps)
    
    target_gradients = np.array([
        [0.00011989, 0.00036228, 0.00027342],
        [0.00018771, 0.00036779, 0.00019101]
        ])
    assert np.allclose(test_gradients, target_gradients)
    print("Success")
    
    
if testing:
    test_function()
    exit()
    
main()