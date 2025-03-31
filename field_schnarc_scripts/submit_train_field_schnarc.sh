#!/bin/bash

TRAIN_SCRIPT="train_field_schnarc.sh"

# Default number of epochs
EPOCHS=100

# Model output directory, get overwritten if --overwrite is set in the TRAIN_SCRIPT
MODEL_PATH="test"

# Default number of submissions
NUM_SUBMISSIONS=1

# Default job name
job_name=$(basename $PWD)

print_usage() {
    echo "Usage: $0 [-n number_of_submissions] [-e number_of_epochs] [-d database] [-m model_path]" >&2
}

# Parse arguments
while getopts "n:e:d:m:" flag
do
    case $flag in
        e) EPOCHS=${OPTARG};;
        d) DATABASE=${OPTARG};;
        m) MODEL_PATH=${OPTARG};;
        n) NUM_SUBMISSIONS=$OPTARG ;;
        *) print_usage; exit 1 ;;
    esac
done

# Input validation
if [ $NUM_SUBMISSIONS -lt 1 ]
then
    echo "Number of submissions must be at least 1" >&2
    exit 1
fi

if [ $NUM_SUBMISSIONS -gt 10 ]
then
    echo "Number of submissions must be at most 10" >&2
    exit 1
fi

# Submit jobs
database=$(realpath $DATABASE)
echo "Database: $database"
echo "Number of epochs: $EPOCHS"
echo "Model path: $MODEL_PATH"
#echo "Number of submissions: $NUM_SUBMISSIONS"

if [ $NUM_SUBMISSIONS -eq 1 ]
then
    sbatch --job-name=$job_name $TRAIN_SCRIPT -e $EPOCHS -d $database -m $MODEL_PATH
    exit 0
else
    echo "Number of submissions must be 1" >&2
    exit 1
fi

# for ((i=0; i<NUM_SUBMISSIONS; i++))
# do
#     submission_dir="model_$i"
#     split_data_folder="$data_folder/split_$i"
#     mkdir -p $submission_dir
#     cd $submission_dir
#     sbatch --job-name="${job_name}_$i" $TRAIN_SCRIPT -e $EPOCHS -d $database
#     cd ..
# done