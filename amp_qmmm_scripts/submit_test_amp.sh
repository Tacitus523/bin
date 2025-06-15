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

if [[ $1 == "--help" || $1 == "-h" ]]; then
    print_usage
fi

# check if the test_amp.py command is available
if ! command -v test_amp.py &> /dev/null
then
    echo "test_amp.py could not be found. Please activate the appropiate environment."
    exit 1
fi

submission_script=$(realpath "$(dirname "$(readlink -f "$0")")/../submit_python_file_justus.sh")
amp_test=$(which test_amp.py)
amp_plot=$(which AMPPlot.py)
if [ -z "$submission_script" ]; then
    echo "Error: Could not find the submission script."
    exit 1
fi
if [ -z "$amp_test" ]; then
    echo "Error: Could not find the amp script test_amp.py."
    exit 1
fi
if [ -z "$amp_plot" ]; then
    echo "Error: Could not find the amp script AMPPlot.py."
    exit 1
fi

if [ -f "train.out" ]; then
    rm train.out
fi

if [ -f "train.err" ]; then
    rm train.err
fi

output=$($submission_script -p $amp_test -c $results_dir)
echo "$output"
job_id=$(echo "$output" | tail -n 1)

cd $results_dir
sbatch --dependency=afterok:$job_id --kill-on-invalid-dep=yes $amp_plot -g amp_qmmm_geoms.extxyz # name hardcoded in the amp_test script
cd -