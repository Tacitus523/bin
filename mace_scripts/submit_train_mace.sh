#!/bin/bash

TRAIN_SCRIPT="train_mace.sh"

CONFIG_FILE="config.yaml"

# Default number of submissions
NUM_SUBMISSIONS=1

# Default job name
job_name=$(basename $PWD)
email_flag="--mail-user=$MY_EMAIL --mail-type=END,FAIL"

print_usage() {
    echo "Usage: $0 [-n number_of_submissions] [-d data_folder] [-c config_file]" >&2
    echo "  -n number_of_submissions : Number of model training submissions (1-10). Default is 1." >&2
    echo "  -d data_folder           : Path to data folder. Required if number_of_submissions > 1." >&2
    echo "  -c config_file           : Path to configuration file. Default is 'config.yaml'." >&2
}

# Parse arguments
while getopts "n:e:d:c:" flag
do
    case $flag in
        c) CONFIG_FILE=${OPTARG} ;;
        e) echo "option -e is deprecated, enter number of epochs directly in config file" ;;
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

config_file=$(readlink -f $CONFIG_FILE)
if [ ! -f "$config_file" ]
then
    echo "Config file does not exist: $config_file" >&2
    exit 1
fi

if [ -n "$DATA_FOLDER" ]
then
    if [ ! -d "$DATA_FOLDER" ]
    then
        echo "Data folder does not exist: $DATA_FOLDER" >&2
        exit 1
    fi
    data_folder=$(readlink -f $DATA_FOLDER)
    data_folder_flag="-d $data_folder"
    echo "Data folder: $data_folder"
fi

echo "Config file: $config_file"
echo "Number of submissions: $NUM_SUBMISSIONS"

if [ $NUM_SUBMISSIONS -eq 1 ]
then
    sbatch --job-name=$job_name $email_flag $TRAIN_SCRIPT $data_folder_flag -c $config_file
    exit 0
fi

if [ -z "$DATA_FOLDER" ]
then
    echo "Data folder is required for multiple submissions" >&2
    exit 1
fi

for ((i=0; i<NUM_SUBMISSIONS; i++))
do
    if [ -n "$data_folder" ]
    then
        split_data_folder="$data_folder/split_$i"
        if [ ! -d "$split_data_folder" ]
        then
            echo "Data folder does not exist: $split_data_folder" >&2
            exit 1
        fi
        split_data_folder_flag="-d $split_data_folder"
    fi

    submission_dir="model_$i"
    mkdir -p $submission_dir
    cd $submission_dir
    sbatch --job-name="${job_name}_$i" $email_flag $TRAIN_SCRIPT $split_data_folder_flag -c $config_file
    cd ..
done