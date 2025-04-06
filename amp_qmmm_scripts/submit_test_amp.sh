#!/usr/bin/env bash
# Submission script for testing a model using the test_amp.py script and submit_python_file_justus.sh script.

print_usage() {
    echo "Usage: $0 <results_dir>"
    echo "Submits a testing job using the specified results directory."
    exit 1
}

results_dir=$1
if [ -z "$results_dir" ]; then
    echo "Error: No results directory provided."
    print_usage
fi

# check if the test_amp.py command is available
if ! command -v test_amp.py &> /dev/null
then
    echo "test_amp.py could not be found. Please activate the appropiate environment."
    exit 1
fi

submission_script=$(realpath "$(dirname "$(readlink -f "$0")")/../submit_python_file_justus.sh")
amp_script=$(which test_amp.py)


$submission_script -p $amp_script -c $results_dir 