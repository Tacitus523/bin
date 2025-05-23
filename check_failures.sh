#!/bin/bash
#Give folder-prefix as $1 and file-prefix as $2
print_usage() {
    echo "Usage: $0 <folder_prefix> <file_prefix>"
    echo "Example: $0 GEOM_06342 sp"
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

# Check if at least one file exists
if [ -z "$(ls $1*/$2*.out 2> /dev/null)" ]; then
    echo "No files found matching $1*/$2*.out"
    exit 1
fi

# Define check function
check_folder() {
    local folder="$1"
    local file_prefix="$2"
    if ! tac ${folder}/${file_prefix}*.out 2> /dev/null | grep -q -m 1 "****ORCA TERMINATED NORMALLY****" 2> /dev/null; then 
        echo "${folder} failed"
    fi
}
export -f check_folder

# Process all folders in parallel
find "$1"* -type d | parallel -j 10 "check_folder {} '$2'"
