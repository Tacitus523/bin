#!/usr/bin/python3
import os

data_lst = ["B3LYP_aug-cc-pVTZ_water"]

experiment_names = ["second_gen", "fourth_gen_mull", "fourth_gen_hirsh", "fourth_gen_hirsh_no_esp"]   
generation_lst = [2, 4, 4, 4]
charge_lst = [[None], ["charges_mull.txt"], ["charges_hirsh.txt"], ["charges_hirsh.txt"]]
esp_lst = [[None], ["esps_by_mm.txt"], ["esps_by_mm.txt"], [None]]

assert all((len(generation_lst), len(charge_lst), len(esp_lst)== len(experiment_names))) , "All Experiments need names and data"
assert len(experiment_names) == len(set(experiment_names)), "Experiment names must be unique"

import_string = """
import sys
sys.path.append('/home/lpetersen/dftb-nn')
from Behler_4th_Gen import run_Behler
"""

summarize_and_run_string = """
file_dict = {
"data_dir": DATA_DIR_NAME,
"geom_data": GEOM_DATA_NAMES,
"energy_data": ENERGY_DATA_NAMES,
"charges_data": CHARGES_DATA_NAMES,
"esp_data": ESP_DATA_NAMES
}

run_Behler(file_dict, GENERATION, DO_TRAINING, SAVE_DIR_NAME)
"""

for DATA_DIR_NAME in data_lst:
    try:
        os.chdir(DATA_DIR_NAME)
    except:
        print("All data_dirs must exist. This is supposed to prevent submissions from wrong folder")
        exit()
    
    for i in range(len(experiment_names)):
        try:
            os.mkdir(experiment_names[i]) #  raises an error if folder exists
            os.chdir(experiment_names[i])
        except:
            print(f"Skipped {experiment_names[i]}, experiment already exists")
            continue

        with open("submission.py", "w") as f:
            f.write(import_string)
            f.write(f'GENERATION = {generation_lst[i]}' + "\n")
            f.write('DO_TRAINING = True' + "\n")
            f.write(f'DATA_DIR_NAME = "{DATA_DIR_NAME}"' + "\n")
            f.write('GEOM_DATA_NAMES = ["geoms.xyz"]' + "\n")
            f.write('ENERGY_DATA_NAMES = ["energy_diff.txt"]' + "\n")
            f.write(f'CHARGES_DATA_NAMES = {charge_lst[i]}' + "\n")
            f.write(f'ESP_DATA_NAMES = {esp_lst[i]}' + "\n")
            f.write('SAVE_DIR_NAME = ""' + "\n")
            f.write(summarize_and_run_string)
            
            os.system(f"qsub -N {experiment_names[i]} /home/lpetersen/bin/qpython.sh submission.py")
            os.system(f"echo $PWD >> /data/lpetersen/checklist.txt")
        os.chdir("..")
    os.chdir("..")
            