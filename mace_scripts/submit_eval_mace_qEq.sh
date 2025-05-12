#!/bin/bash
#SBATCH --job-name=eval_mace_qEq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=eval.out
#SBATCH --error=eval.err
#SBATCH --gres=gpu:1

DATA_FOLDER="/lustre/work/ws/ws1/ka_he8978-dipeptide/training_data/B3LYP_aug-cc-pVTZ_vacuum"
TEST_FILE="test.extxyz"
MODEL_FILE="QEq_swa.model"
OUTPUT_FILE="geoms_mace.extxyz"

print_usage() {
    echo "Usage: $0 [-d data_folder]" >&2
}

while getopts ":d:m:" flag
do
    case "${flag}" in
        d) DATA_FOLDER=${OPTARG};;
        m) MODEL_FILE=${OPTARG};;
        *) print_usage; exit 1;;
    esac
done

test_file=$(readlink -f $DATA_FOLDER/$TEST_FILE)
if [ ! -f "$test_file" ]
then
    echo "Test file not found: $test_file" >&2
    exit 1
fi

model_file=$MODEL_FILE
if [ ! -f "$model_file" ]
then
    echo "Model file not found: $model_file" >&2
    exit 1
fi

PLOT_SCRIPT=$(which MacePlot.py)
if [ -z "$PLOT_SCRIPT" ]
then
    echo "MacePlot.py not found in PATH. Please check your environment." >&2
    exit 1
fi

echo "Using test file: $test_file"
echo "Using model file: $model_file"

# Check if the script is being run in a SLURM job
# If not, submit it as a job
# and set up the job dependency for the plot script
if [ -z "$SLURM_JOB_ID" ]
then
    eval_output=$(sbatch --parsable $0 -d $DATA_FOLDER -m $model_file)
    echo "Submitted evaluation job with ID: $eval_output"
    plot_output=$(sbatch --dependency=afterok:$eval_output --kill-on-invalid-dep=yes --parsable $PLOT_SCRIPT -g $OUTPUT_FILE)
    echo "Submitted plot job with ID: $plot_output"
    exit
fi

export PYTHONPATH=${PYTHONPATH}:/lustre/home/ka/ka_ipc/ka_he8978/MACE_QEq_development/mace-tools
export PYTHONPATH=${PYTHONPATH}:/lustre/home/ka/ka_ipc/ka_he8978/MACE_QEq_development/graph_longrange

echo "Starting evaluation on $SLURM_JOB_NODELIST: $(date)"

python /lustre/home/ka/ka_ipc/ka_he8978/MACE_QEq_development/mace-tools/scripts/eval_qeq.py \
        --configs="$test_file" \
        --model="$model_file" \
        --output="$OUTPUT_FILE" \
        --device="cuda" 