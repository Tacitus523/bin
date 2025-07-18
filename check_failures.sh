#!/bin/bash
#Give folder-prefix as $1 and file-prefix as $2
print_usage() {
    echo "Usage: $0 <folder_prefix> <file_prefix>"
    echo "DFT Example: $0 GEOM_ sp"
    echo "DFTB Example: $0 GEOM_ detailed.out"
}

if [ $1 = "-h" ] || [ $1 = "--help" ]; then
    print_usage
    exit 0
fi
if [ $# -ne 2 ]; then
    echo "Error: Invalid number of arguments."
    print_usage
    exit 1
fi

set -o errexit   # (or set -e) cause batch script to exit immediately when a command fails.

folder_prefix=$1
# cut .out suffix from file_prefix
file_prefix=${2%.out}

# Check if at least one file exists
if [ -z "$(ls $folder_prefix*/$file_prefix*.out 2> /dev/null)" ]; then
    echo "No files found matching $folder_prefix*/$file_prefix*.out"
    exit 1
fi

# Define check function
function check_DFT_folder() {
    local folder="$1"
    local file_prefix="$2"
    if ! tac ${folder}/${file_prefix}*.out 2> /dev/null | grep -q -m 1 "****ORCA TERMINATED NORMALLY****" 2> /dev/null; then 
        echo "${folder} failed"
    fi
}
export -f check_DFT_folder

function check_DFTB_folder() {
    local folder="$1"
    local file_prefix="$2"
    if ! tac ${folder}/${file_prefix}*.out 2> /dev/null | grep -q -m 1 "SCC converged" 2> /dev/null; then 
        echo "${folder} failed"
    fi
}
export -f check_DFTB_folder

# Process all folders in parallel
if [[ $(basename $file_prefix) == "detailed" ]] 
then 
    echo "Checking DFTB folders for convergence..."
    find "$folder_prefix"* -type d | parallel -j 10 "check_DFTB_folder {} '$file_prefix'"
else
    echo "Checking DFT folders for convergence..."
    find "$folder_prefix"* -type d | parallel -j 10 "check_DFT_folder {} '$file_prefix'"
fi
