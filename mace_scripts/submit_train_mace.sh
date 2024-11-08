#!/bin/bash

TRAIN_SCRIPT="/lustre/home/ka/ka_ipc/ka_he8978/bin/mace_scripts/train_mace.sh"

# Default number of submissions
num_submissions=1

# Default job name
job_name=$(basename $PWD)

print_usage() {
    echo "Usage: $0 [-n number_of_submissions]" >&2
}

# Parse arguments
while getopts "n:" opt
do
    case $opt in
        n) num_submissions=$OPTARG ;;
        *) print_usage; exit 1 ;;
    esac
done

# Input validation
if [ $num_submissions -lt 1 ]
then
    echo "Number of submissions must be at least 1" >&2
    exit 1
fi

if [ $num_submissions -gt 10 ]
then
    echo "Number of submissions must be at most 10" >&2
    exit 1
fi

# Submit jobs
if [ $num_submissions -eq 1 ]
then
    sbatch --job-name=$job_name $TRAIN_SCRIPT
    exit 0
fi

for ((i=0; i<num_submissions; i++))
do
    submission_dir="model_$i"
    mkdir -p $submission_dir
    cd $submission_dir
    sbatch --job-name="${job_name}_$i" $TRAIN_SCRIPT
    cd ..
done