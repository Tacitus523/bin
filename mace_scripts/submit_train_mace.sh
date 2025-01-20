#!/bin/bash

TRAIN_SCRIPT="train_mace.sh"

# Default data folder
DATA_FOLDER="/lustre/work/ws/ws1/ka_he8978-dipeptide/training_data/B3LYP_aug-cc-pVTZ_vacuum"
#DATA_FOLDER="/lustre/work/ws/ws1/ka_he8978-thiol_disulfide/training_data/B3LYP_aug-cc-pVTZ_vacuum"
#DATA_FOLDER="/lustre/work/ws/ws1/ka_he8978-thiol_disulfide/training_data/B3LYP_aug-cc-pVTZ_water"

# Default number of epochs
EPOCHS=100

# Default number of submissions
NUM_SUBMISSIONS=1

# Default job name
job_name=$(basename $PWD)

print_usage() {
    echo "Usage: $0 [-n number_of_submissions] [-e number_of_epochs] [-d data_folder]" >&2
}

# Parse arguments
while getopts "n:e:d:" flag
do
    case $flag in
        e) EPOCHS=${OPTARG};;
        d) DATA_FOLDER=${OPTARG};;
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
data_folder=$(realpath $DATA_FOLDER)
echo "Data folder: $data_folder"
echo "Number of epochs: $EPOCHS"
echo "Number of submissions: $NUM_SUBMISSIONS"

if [ $NUM_SUBMISSIONS -eq 1 ]
then
    sbatch --job-name=$job_name $TRAIN_SCRIPT -e $EPOCHS -d $data_folder
    exit 0
fi

for ((i=0; i<NUM_SUBMISSIONS; i++))
do
    submission_dir="model_$i"
    split_data_folder="$data_folder/split_$i"
    mkdir -p $submission_dir
    cd $submission_dir
    sbatch --job-name="${job_name}_$i" $TRAIN_SCRIPT -e $EPOCHS -d $data_folder
    cd ..
done