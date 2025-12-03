#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=train.out
#SBATCH --error=train.err
#SBATCH --gres=gpu:1

module load compiler/gnu/10.2
module load devel/cuda/12.3
module load lib/cudnn/9.0.0_cuda-12.3

export PYTHONPATH=${PYTHONPATH}:"/lustre/home/ka/ka_ipc/ka_he8978/MACE_QEq_development/mace-tools"
export PYTHONPATH=${PYTHONPATH}:"/lustre/home/ka/ka_ipc/ka_he8978/MACE_QEq_development/graph_longrange"

# Relative paths in the data folder
TRAIN_FILE="train.extxyz"
VALID_FILE="valid.extxyz"
TEST_FILE="test.extxyz"

EVAL_SCRIPT="$(which submit_eval_mace_qEq.sh)"

# Atomization energies for the DFT and DFTB methods, deprecated since config.yaml
DFT_E0s='{1: -13.575035506869515, 6: -1029.6173622986487, 7: -1485.1410643783852, 8: -2042.617308911902, 16: -10832.265333248919}'
DFTB_E0s='{1: -7.192493802609272, 6: -42.8033008522276, 7: -65.55164277599535, 8: -94.82677849249036}'
DELTA_E0s='{1: -5.965049432479681, 6: -990.3248623363888, 7: -1424.8147941575799, 8: -1957.1200122411778, 16: -10769.044186224903}'

print_usage() {
    echo "Usage: $0 -c <config_file>" 1>&2
    echo "  -c <config_file>   Path to the YAML configuration file for training" 1>&2
}

while getopts c: flag
do
    case "${flag}" in
        c) config_file=${OPTARG};;
        *) print_usage; exit 1;;
    esac
done

if [ -z "$config_file" ]
then
    echo "Error: Configuration file is required." >&2
    print_usage
    exit 1
fi

if ! [ -f "$config_file" ]
then
    echo "Error: Configuration file not found: $config_file" >&2
    exit 1
fi

model_name=$(yq e '.name' $config_file)

if [ -z $SLURM_JOB_NAME ]; then
    WANDB_NAME="${model_name}_$(basename $PWD)"
else
    WANDB_NAME="${model_name}_${SLURM_JOB_NAME}"
fi

echo "Starting training: $(date)"

python /lustre/home/ka/ka_ipc/ka_he8978/MACE_QEq_development/mace-tools/scripts/lukas_train.py  \
    --config $config_file \
    --seed=$RANDOM \
    --wandb_name=$WANDB_NAME

training_exit_status=$?

if [ $training_exit_status -ne 0 ]
then
    echo "Training failed with exit status $training_exit_status" >&2
    exit $training_exit_status
else
    echo "Training completed successfully: $(date)"
fi

# Evaluate the trained model
if ! [ -x "$EVAL_SCRIPT" ]
then
    echo "Evaluation script not found or not executable: $EVAL_SCRIPT" >&2
    exit 1
fi

model_name=$(yq e '.name' $config_file)
model_file="${model_name}_swa_compiled.pt"
if ! [ -f "$model_file" ]
then
    echo "SWA model file not found: $model_file" >&2
    echo "Falling back to regular model file." >&2
    model_file="${model_name}_compiled.pt"
fi
if ! [ -f "$model_file" ]
then
    echo "Model file not found: $model_file" >&2
    echo "Skipping evaluation." >&2
    exit 1
fi

test_file=$(yq eval '.test_file' $config_file)
data_folder=$(dirname $test_file)
$EVAL_SCRIPT -m $model_file -d $data_folder
