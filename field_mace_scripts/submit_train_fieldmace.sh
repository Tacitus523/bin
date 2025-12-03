#!/bin/bash

TRAIN_SCRIPT="train_fieldmace.sh"

CONFIG_FILE="config.yaml"
TRAIN_FILE="train.xyz"
VALID_FILE="valid.xyz"
TEST_FILE="test.xyz"

# Default number of submissions
NUM_SUBMISSIONS=1

# Default job name
job_name=$(basename $PWD)
email_flag="--mail-user=$MY_EMAIL --mail-type=END,FAIL"

print_usage() {
    echo "Usage: $0 [-n number_of_submissions] [-d data_folder] [-c config_file]" >&2
    echo "  -n number_of_submissions : Number of model training submissions (1-10). Default is 1." >&2
    echo "  -d data_folder           : Path to data folder. Optional if config is given" >&2
    echo "  -c config_file           : Path to configuration file. Default is 'config.yaml'." >&2
}

# Parse arguments
while getopts "n:d:c:" flag
do
    case $flag in
        c) CONFIG_FILE=${OPTARG} ;;
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
    yq eval '.train_file = $train_file | .valid_file = $valid_file | .test_file = $test_file' --arg train_file "$data_folder/$TRAIN_FILE" --arg valid_file "$data_folder/$VALID_FILE" --arg test_file "$data_folder/$TEST_FILE" $config_file > temp_config.yaml
    mv temp_config.yaml $config_file
else
    data_folder=$(yq eval '.train_file' $config_file)
fi


echo "Config file: $config_file"
echo "Number of submissions: $NUM_SUBMISSIONS"

if [ $NUM_SUBMISSIONS -eq 1 ]
then
    sbatch --job-name=$job_name $email_flag $TRAIN_SCRIPT -c $config_file
    exit 0
fi

for ((i=0; i<NUM_SUBMISSIONS; i++))
do
    submission_dir="model_$i"
    split_config_file="config.yaml"
    if [ -n "$data_folder" ]
    then
        split_data_folder="$data_folder/split_$i"
        if [ ! -d "$split_data_folder" ]
        then
            echo "Data folder does not exist: $split_data_folder" >&2
            exit 1
        fi
        mkdir -p $submission_dir
        yq eval '.train_file = "'"$split_data_folder/$TRAIN_FILE"'" | .valid_file = "'"$split_data_folder/$VALID_FILE"'" | .test_file = "'"$split_data_folder/$TEST_FILE"'"' "$config_file" > $submission_dir/$split_config_file
    else
        echo "Data folder is required for multiple submissions" >&2
        exit 1
    fi
    
    mkdir -p $submission_dir
    cd $submission_dir
    sbatch --job-name="${job_name}_$i" $email_flag $TRAIN_SCRIPT -c $split_config_file 
    cd ..
done