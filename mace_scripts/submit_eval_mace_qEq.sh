#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=eval.out
#SBATCH --error=eval.err
#SBATCH --gres=gpu:1

#DATA_FOLDER="/lustre/work/ws/ws1/ka_he8978-dipeptide/training_data/B3LYP_aug-cc-pVTZ_vacuum"
DATA_FOLDER="/lustre/work/ws/ws1/ka_he8978-dipeptide/training_data/B3LYP_aug-cc-pVTZ_water"
#DATA_FOLDER="/lustre/work/ws/ws1/ka_he8978-thiol_disulfide/training_data/B3LYP_aug-cc-pVTZ_water"
#DATA_FOLDER="/lustre/work/ws/ws1/ka_he8978-thiol_disulfide/training_data/B3LYP_aug-cc-pVTZ_vacuum"
#DATA_FOLDER="/lustre/work/ws/ws1/ka_he8978-fr0/training_data/DFTB_OB2-1-1_DMSO"
TEST_FILE="$DATA_FOLDER/test.extxyz"
OUTPUT_FILE="geoms_mace.extxyz"

PLOT_SCRIPT=$(which MacePlot.py)
if [ -z "$PLOT_SCRIPT" ]; then
    echo "MacePlot.py not found in PATH. Please check your environment."
    exit 1
fi

echo "Using test file: $TEST_FILE"

# Check if the script is being run in a SLURM job
# If not, submit it as a job
# and set up the job dependency for the plot script
if [ -z "$SLURM_JOB_ID" ]; then
    eval_output=$(sbatch --parsable $0)
    echo "Submitted evaluation job with ID: $eval_output"
    plot_output=$(sbatch --dependency=afterok:$eval_output --kill-on-invalid-dep=yes --parsable $PLOT_SCRIPT -g $OUTPUT_FILE)
    echo "Submitted plot job with ID: $plot_output"
    exit
fi

export PYTHONPATH=${PYTHONPATH}:/lustre/home/ka/ka_ipc/ka_he8978/MACE_QEq_development/mace-tools
export PYTHONPATH=${PYTHONPATH}:/lustre/home/ka/ka_ipc/ka_he8978/MACE_QEq_development/graph_longrange

echo "Starting evaluation on $SLURM_JOB_NODELIST: $(date)"

python /lustre/home/ka/ka_ipc/ka_he8978/MACE_QEq_development/mace-tools/scripts/eval_qeq.py \
        --configs="$TEST_FILE" \
        --model="QEq_swa.model" \
        --output="$OUTPUT_FILE" \
        --device="cuda" 