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

parser = argparse.ArgumentParser(
    description = 'Transforming .xyz-file to .gro-file'
)
parser.add_argument('-f', '--file', help='Input file (.xyz) for transformation.')
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