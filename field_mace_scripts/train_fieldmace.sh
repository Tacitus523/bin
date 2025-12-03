#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --time=120:00:00
#SBATCH --output=train.out
#SBATCH --error=train.err
#SBATCH --gres=gpu:1
#SBATCH --mail-user=lukas.petersen@kit.edu
#SBATCH --mail-type=END,FAIL

module load compiler/gnu/10.2
module load devel/cuda/12.3
module load lib/cudnn/9.0.0_cuda-12.3

# Relative paths in the data folder
TRAIN_FILE="train.xyz"
VALID_FILE="valid.xyz"
TEST_FILE="test.xyz"
EVAL_SCRIPT="$(which submit_eval_fieldmace.sh)"
# Atomization energies for the DFT and DFTB methods, deprecated since config.yaml
DFT_E0s='{1: -13.575035506869515, 6: -1029.6173622986487, 7: -1485.1410643783852, 8: -2042.617308911902, 16: -10832.265333248919}'
DFTB_E0s='{1: -7.192493802609272, 6: -42.8033008522276, 7: -65.55164277599535, 8: -94.82677849249036}'


if [ -z $SLURM_JOB_NAME ]; then
    WANDB_NAME=$(basename $PWD)
else
    WANDB_NAME=$SLURM_JOB_NAME
fi

print_usage() {
    echo "Usage: $0 [-c config_file]" >&2
    echo "  -c config_file : Path to configuration file. Default is 'config.yaml'." >&2
}

while getopts c: flag
do
    case "${flag}" in
        c) config_file=${OPTARG};;
        *) print_usage; exit 1;;
    esac
done

echo "Starting training: $(date)"

mace_run_train \
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

# # Convert the model to a scripted model
# if [ $training_exit_status -eq 0 ]
# then
#     convert_model_to_scripted_model.py --model_prefix $MODEL_NAME
# fi

# Evaluate the trained model
if ! [ -x "$EVAL_SCRIPT" ]
then
    echo "Evaluation script not found or not executable: $EVAL_SCRIPT" >&2
    exit 1
fi

model_name=$(yq e '.name' $config_file)
model_file="${model_name}_stagetwo_compiled.model"
if ! [ -f "$model_file" ]
then
    echo "SWA model file not found: $model_file" >&2
    echo "Falling back to regular model file." >&2
    model_file="${model_name}_compiled.model"
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
