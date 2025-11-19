#!/bin/bash
#Give folder-prefix as $1 and file-prefix as $2
print_usage() {
    echo "Usage: $0 <folder_prefix> <file_prefix>"
    echo "DFT Example: $0 GEOM_ sp"
    echo "DFTB Example: $0 GEOM_ detailed.out"
}

if [ $1 = "-h" ] || [ $1 = "--help" ]
then
    print_usage
    exit 0
fi
if [ $# -ne 2 ]
then
    echo "Error: Invalid number of arguments."
    print_usage
    exit 1
fi

set -o errexit   # (or set -e) cause batch script to exit immediately when a command fails.

folder_prefix=$1
# cut .out suffix from file_prefix
file_prefix=${2%.out}
folder_prefix_dirname=$(dirname "$folder_prefix")
folder_prefix_basename=$(basename "$folder_prefix")

# Check if at least one file exists
if [ -z "$(ls $folder_prefix*/$file_prefix*.out 2> /dev/null)" ]; then
    echo "No files found matching $folder_prefix*/$file_prefix*.out"
    exit 1
fi

# Define check function
function check_DFT_folder() {
    local folder="$1"
    if ! tac ${folder}/${file_prefix}*.out 2> /dev/null | grep -q -m 1 "****ORCA TERMINATED NORMALLY****" 2> /dev/null
    then 
        echo "${folder} failed"
    fi
}
export -f check_DFT_folder

function check_DFTB_folder() {
    local folder="$1"
    if ! grep -q -m 1 "SCC converged" ${folder}/${file_prefix}*.out 2> /dev/null
    then 
        echo "${folder} failed"
    fi
}
export -f check_DFTB_folder

folders=$(find "$folder_prefix_dirname" -maxdepth 1 -name "${folder_prefix_basename}*" \( -type d -o -type l \) | sort -V) # Ensures numerical ordering without padded folders --> folder_0, folder_1, folder_2, ... instead of folder_0, folder_1, folder_10, ...

# Process all folders in parallel
# If the file prefix is "detailed", we assume DFTB calculations (since DFTB output files are named "detailed.out")
export file_prefix
if [[ $(basename $file_prefix) == "detailed" ]] 
then 
    parallel -j 32 --env file_prefix "check_DFTB_folder {}" ::: $folders
else
    parallel -j 32 --env file_prefix "check_DFT_folder {}" ::: $folders
fi
