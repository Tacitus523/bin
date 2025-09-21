#!/usr/bin/env bash
print_usage() {
    echo "Usage: $0 <input_cub1> <input_cub2>"
    echo "This script runs Multiwfn on the specified .cub files to calculate the grid difference."
}

script_path=$(readlink -f "$0")
script_folder=$(dirname "$script_path")
output_cub_diff="grid_diff.cub"


if [ "$#" -ne 2 ]; then
    print_usage
    exit 1
fi

if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    print_usage
    exit 0
fi

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: Both input .cub files must be specified."
    print_usage
    exit 1
fi

input_cub1="$1"
input_cub2="$2"
multiwfn_input="13\n11\n4\n${input_cub2}\n0\n${output_cub_diff}\n-1\nq"

if ! [ -f "$input_cub1" ]; then
    echo "Input file '$input_cub1' does not exist."
    exit 1
fi

if ! [ -f "$input_cub2" ]; then
    echo "Input file '$input_cub2' does not exist."
    exit 1
fi

if ! which Multiwfn > /dev/null; then
    echo "Multiwfn is not installed or not in PATH."
    exit 1
fi

echo -e "$multiwfn_input" | Multiwfn "$input_cub1"
if [ $? -ne 0 ]; then
    echo "Multiwfn execution failed."
    exit 1
fi
echo "Multiwfn difference calculation completed successfully."

