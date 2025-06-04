#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=train.out
#SBATCH --error=train.err
#SBATCH --gres=gpu:1
#SBATCH --mail-user=lukas.petersen@kit.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --oversubscribe # Allow sharing of resources

module load compiler/gnu/10.2
module load devel/cuda/12.3
module load lib/cudnn/9.0.0_cuda-12.3

export PYTHONPATH=${PYTHONPATH}:"/lustre/home/ka/ka_ipc/ka_he8978/MACE_QEq_development/mace-tools"
export PYTHONPATH=${PYTHONPATH}:"/lustre/home/ka/ka_ipc/ka_he8978/MACE_QEq_development/graph_longrange"

# Relative paths in the data folder
TRAIN_FILE="train.extxyz"
VALID_FILE="valid.extxyz"
TEST_FILE="test.extxyz"
# Default value, gets overwritten by the command line argument from submit_train_mace.sh
EPOCHS=100

MODEL_NAME="QEq"

# Atomization energies for the DFT and DFTB methods, deprecated since config.yaml
DFT_E0s='{1: -13.575035506869515, 6: -1029.6173622986487, 7: -1485.1410643783852, 8: -2042.617308911902, 16: -10832.265333248919}'
DFTB_E0s='{1: -7.192493802609272, 6: -42.8033008522276, 7: -65.55164277599535, 8: -94.82677849249036}'

if [ -z $SLURM_JOB_NAME ]; then
    WANDB_NAME=$(basename $PWD)
else
    WANDB_NAME=$SLURM_JOB_NAME
fi

print_usage() {
    echo "Usage: $0 [-e number_of_epochs] [-d data_folder]" >&2
}

# Parse command line arguments for epochs and data folder
while getopts e:d:c: flag
do
    case "${flag}" in
        e) EPOCHS=${OPTARG};;
        d) DATA_FOLDER=${OPTARG};;
        c) config_file=${OPTARG};;

        *) print_usage; exit 1;;
    esac
done

data_folder=$(readlink -f $DATA_FOLDER)
train_file=$(readlink -f $DATA_FOLDER/$TRAIN_FILE)
valid_file=$(readlink -f $DATA_FOLDER/$VALID_FILE)
test_file=$(readlink -f $DATA_FOLDER/$TEST_FILE)

echo "Starting training: $(date)"
echo "Data folder: $data_folder"
echo "Train file: $train_file"
echo "Valid file: $valid_file"
echo "Test file: $test_file"
echo "Model name: $MODEL_NAME"
echo "Number of epochs: $EPOCHS"

python /lustre/home/ka/ka_ipc/ka_he8978/MACE_QEq_development/mace-tools/scripts/lukas_train.py  \
    --config $config_file \
    --name=$MODEL_NAME \
    --seed=$RANDOM \
    --train_file=$train_file \
    --valid_file=$valid_file \
    --test_file=$test_file \
    --max_num_epochs=$EPOCHS \
    --wandb_name=$WANDB_NAME \

training_exit_status=$?

echo "Finished training: $(date)"

# # Convert the model to a scripted model
# if [ $training_exit_status -eq 0 ]
# then
#     convert_model_to_scripted_model.py --model_prefix $MODEL_NAME
# fi
