#!/usr/bin/python3
import os
import sys
import re
import argparse

REPLACED_RESID_NAMES = [
    "1ACE",
    "1ACE",
    "1ACE",
    "1ACE",
    "1ACE",
    "1ACE",
    "2ALA",
    "2ALA",
    "2ALA",
    "2ALA",
    "2ALA",
    "2ALA",
    "2ALA",
    "2ALA",
    "2ALA",
    "2ALA",
    "3NME",
    "3NME",
    "3NME",
    "3NME",
    "3NME",
    "3NME"
]


#REPLACED_RESID_NAMES = ["1ADE"]*15
#REPLACED_RESID_NAMES = ["1CYT"]*13
#REPLACED_RESID_NAMES = ["1GUA"]*16
#REPLACED_RESID_NAMES = ["1THY"]*15
#REPLACED_RESID_NAMES = ["1URA"]*12

parser = argparse.ArgumentParser(
    description = 'Transforming .xyz-file to .gro-file'
)
parser.add_argument('file', help='Input file (.xyz) for transformation.')
args = parser.parse_args()
input_file = args.file

# convert .xyz to .gro
os.system(f"/usr/local/run/openbabel-2.4.1/bin/obabel -ixyz {input_file} -O geom.gro")
# Replace residue names and atom types to fit .gro format
with open("geom.gro","r") as f:
	data_lines = f.readlines()

data=""
data += data_lines[0]
data += data_lines[1]
for idx, line in enumerate(data_lines[2:-1]):
	data_line = re.sub(r"\d(UN[KL]|GLY)", REPLACED_RESID_NAMES[idx], line, 1)
	data += data_line
data += data_lines[-1]

with open("geom.gro","w") as f:
	f.write(data)