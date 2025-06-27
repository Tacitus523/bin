#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=eval.out
#SBATCH --error=eval.err
#SBATCH --gres=gpu:1

DATA_FOLDER="/lustre/work/ws/ws1/ka_he8978-fr0/training_data/DFTB_OB2-1-1_all_solvents/field_mace"
MODEL_FILE="FieldMace_stagetwo.model"
TEST_FILE="test.xyz"
OUTPUT_FILE="geoms_fieldmace.xyz"

print_usage() {
    echo "Usage: $0 [-d data_folder] [-m model_file]" >&2
    echo "  -d data_folder   Path to the folder containing the test file (default: $DATA_FOLDER)" >&2
    echo "  -m model_file     Path to the model file (default: $MODEL_FILE)" >&2
    echo "Example: $0 -d /path/to/data -m /path/to/model.model" >&2
}

while getopts ":d:m:" flag 
do
    case "${flag}" in
        d) DATA_FOLDER=${OPTARG};;
        m) MODEL_FILE=${OPTARG};;
        *) print_usage; exit 1;;
    esac
done

test_file=$(realpath $DATA_FOLDER/$TEST_FILE)
model_file=$MODEL_FILE

# Check if the data folder is set and test file exists
if [ -z "$DATA_FOLDER" ]; then
    echo "Data folder not specified. Please provide a data folder."
    exit 1
fi
if [ ! -f "$test_file" ]; then
    echo "Test file not found: $test_file"
    exit 1
fi

EVAL_SCRIPT=$(which mace_eval_configs)
if [ -z "$EVAL_SCRIPT" ]; then
    echo "mace_eval_configs not found in PATH. Please check your environment."
    exit 1
fi

PLOT_SCRIPT=$(which FieldMacePlot.py)
if [ -z "$PLOT_SCRIPT" ]; then
    echo "MacePlot.py not found in PATH. Please check your environment."
    exit 1
fi

echo "Using test file: $test_file"
echo "Using output file: $OUTPUT_FILE"

# Check if the script is being run in a SLURM job
# If not, submit it as a job
# and set up the job dependency for the plot script
if [ -z "$SLURM_JOB_ID" ]; then
    eval_output=$(sbatch --parsable $0 -d $DATA_FOLDER -m $model_file)
    echo "Submitted evaluation job with ID: $eval_output"
    plot_output=$(sbatch --dependency=afterok:$eval_output --kill-on-invalid-dep=yes --parsable $PLOT_SCRIPT -g $OUTPUT_FILE)
    echo "Submitted plot job with ID: $plot_output"
    exit
fi

echo "Starting evaluation on $SLURM_JOB_NODELIST: $(date)"

$EVAL_SCRIPT \
    --configs="$test_file" \
    --batch_size=8 \
    --model=$model_file \
    --output="$OUTPUT_FILE" \
    --device="cuda" \
    --info_prefix="MACE_"