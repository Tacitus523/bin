#!/usr/bin/env bash

print_usage() {
    echo "Usage: $0 <input_molden_input> <total_charge>"
    echo "Example: $0 your_file.molden.input 0"
    echo "This script runs Multiwfn on the specified molden-input('orca_2mkl YOUR_FILE.gbw -molden') file."
}

input_file="$1"
total_charge="$2"
input_base=$(basename "$input_file" .molden.input)
output_prefix="EEM"
output_file="$output_prefix.chg"

script_path=$(readlink -f "$0")
script_folder=$(dirname "$script_path")
multiwfn_input="7\n17\ng\n2\n$total_charge\n0\ny\n-1\n0\nq"

if [ -z "$input_file" ]; then
    print_usage >&2
    exit 1
fi

if [ -z "$total_charge" ]; then
    echo "Total charge not provided." >&2
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

echo -e "$multiwfn_input" | Multiwfn $input_file > ${output_prefix}.log

if [ $? -ne 0 ]; then
    echo "Multiwfn execution for $output_prefix charges failed." >&2
    exit 1
fi
echo "Multiwfn execution for $output_prefix charges completed successfully."

mv $input_base.chg $output_file