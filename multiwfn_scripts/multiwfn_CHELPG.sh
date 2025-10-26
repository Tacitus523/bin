#!/usr/bin/env bash

print_usage() {
    echo "Usage: $0 <input_molden_input>"
    echo "This script runs Multiwfn on the specified molden-input('orca_2mkl YOUR_FILE.gbw -molden') file."
}

input_file="$1"

input_base=$(basename "$input_file" .molden.input)
output_prefix="CHELPG"
output_file="$output_prefix.chg"

script_path=$(readlink -f "$0")
script_folder=$(dirname "$script_path")
multiwfn_input="$script_folder/inputs/$output_prefix.txt"

if [ -z "$input_file" ]; then
    print_usage >&2
    exit 1
fi

if ! [ -f "$input_file" ]; then
    echo "Input file '$input_file' does not exist." >&2
    exit 1
fi

if ! which Multiwfn > /dev/null; then
    echo "Multiwfn is not installed or not in PATH." >&2
    exit 1
fi

Multiwfn $input_file < $multiwfn_input > ${output_prefix}.log

if [ $? -ne 0 ]; then
    echo "Multiwfn execution for $output_prefix charges failed." >&2
    exit 1
fi
echo "Multiwfn execution for $output_prefix charges completed successfully."

mv $input_base.chg $output_file