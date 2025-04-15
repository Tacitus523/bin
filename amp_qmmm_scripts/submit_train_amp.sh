#!/usr/bin/env bash
# Submission script for training a model using the train_amp.py script and submit_python_file_justus.sh script.

print_usage() {
    echo "Usage: $0 <config_file>"
    echo "Submits a training job using the specified config file."
    exit 1
}

config_file=$1
if [ -z "$config_file" ]; then
    echo "Error: No config file provided."
    print_usage
fi

# check if the train_amp.py command is available
if ! command -v train_amp.py &> /dev/null
then
    echo "train_amp.py could not be found. Please activate the appropiate environment."
    exit 1
fi

submission_script=$(realpath "$(dirname "$(readlink -f "$0")")/../submit_python_file_justus.sh")
amp_script=$(which train_amp.py)


$submission_script -p $amp_script -c $config_file -e