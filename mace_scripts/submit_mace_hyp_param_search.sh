#!/bin/bash

#STUDY_NAME="num_cutoff_basis"
MAX_TRIALS=1000
MAX_EPOCHS_SEARCH=67
HYPERBAND_FACTOR=3
MAX_DATASET_SIZE=1000

CONFIG_FILE="config.yaml"
N_TASKS=4
N_GPUS=2


module load compiler/gnu/10.2
module load devel/cuda/12.3
module load lib/cudnn/9.0.0_cuda-12.3

export PYTHONPATH=${PYTHONPATH}:"/lustre/home/ka/ka_ipc/ka_he8978/MACE_QEq_development/mace-tools"
export PYTHONPATH=${PYTHONPATH}:"/lustre/home/ka/ka_ipc/ka_he8978/MACE_QEq_development/graph_longrange"

wandb_name=$(basename $PWD)
wandb_flag="--wandb_name $wandb_name"

submit_python_file_justus.sh -e -j $N_TASKS -g $N_GPUS $HOME/MACE_QEq_development/mace-tools/scripts/lukas_hyperparameter_search.py -- \
    --config $CONFIG_FILE \
    --max_trials $MAX_TRIALS \
    --max_epochs_search $MAX_EPOCHS_SEARCH \
    --max_train_size $MAX_DATASET_SIZE \
    --factor $HYPERBAND_FACTOR \
    --n_jobs $N_TASKS \
    $wandb_flag #\
    #--study_name $STUDY_NAME \
