#!/usr/bin/env bash

print_usage() {
    echo "Usage: $0 <input_gbw>"
    echo "This script runs Multiwfn on the specified gbw or molden-input('orca_2mkl YOUR_FILE.gbw -molden') file."
}

input_file="$1"

script_path=$(readlink -f "$0")
script_folder=$(dirname "$script_path")
multiwfn_input="$script_folder/inputs/ESPpt.txt"

if [ -z "$input_file" ]; then
    print_usage
    exit 1
fi

if ! [ -f "$input_file" ]; then
    echo "Input file '$input_file' does not exist."
    exit 1
fi

if ! which Multiwfn > /dev/null; then
    echo "Multiwfn is not installed or not in PATH."
    exit 1
fi

Multiwfn $input_file < $multiwfn_input

if [ $? -ne 0 ]; then
    echo "Multiwfn execution failed."
    exit 1
fi
echo "Multiwfn execution completed successfully."

input_base=$(basename "$input_file" .gbw)
input_base=$(basename "$input_base" .molden.input)

mv vtx.pdb "${input_base}_vtx.pdb"
mv mol.pdb "${input_base}.pdb"
echo "Output files renamed to ${input_base}_vtx.pdb and ${input_base}.pdb."
echo "You can visualize the output files using VMD or other visualization tools."