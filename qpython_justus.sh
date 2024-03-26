#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=72:00:00
#SBATCH --output=train.out
#SBATCH --error=train.err
#SBATCH --open-mode=append
#SBATCH --gres=gpu:1

PYTHON_ENV=kgcnn_new

pythonfile=$1
config_path="$2" # optional

# Echo important information into file
echo "# Date: " $(date)
echo "# Hostname: " $SLURM_JOB_NODELIST
echo "# Job ID: " $SLURM_JOB_ID

# In case of external API usage I saved some API-keys here
if [ -f ~/.api_keys ]; then
    . ~/.api_keys
fi

# For WandB:
export WANDB_MODE=offline # no internet connection during calculation on nodes

CONDA_HOME=$(dirname $(dirname $CONDA_EXE))
source $CONDA_HOME/etc/profile.d/conda.sh
conda activate $PYTHON_ENV

# OpenMP needs this: set stack size to unlimited
ulimit -s unlimited

echo "Amount of GPUs for job: $SLURM_JOB_GPUS"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"
echo "GPU Ordinal: $GPU_DEVICE_ORDINAL"

if [ -z "$config_path" ]
then time python3 $pythonfile -g 0 # SLURM restricts the visible GPUs itself, so GPU_ID is always 0
else time python3 $pythonfile -g 0 -c $config_path
fi

