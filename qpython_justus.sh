#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --time=200:00:00
#SBATCH --output=train.out
#SBATCH --error=train.err
##SBATCH --open-mode=append

#PYTHON_ENV=amp_qmmm
#PYTHON_ENV=kgcnn_new
PYTHON_ENV=mace_env


pythonfile="$1"
shift  # Remove the first argument (python file) from the list of options
all_opts="$*" # Collect all options in a variable

if [ -z "$pythonfile" ]; then
    echo "Usage: $0 <pythonfile> [options]"
    exit 1
elif [ ! -f "$pythonfile" ]; then
    echo "Error: Python file '$pythonfile' does not exist."
    exit 1
fi

# Echo important information into file
echo "# Date: " $(date)
echo "# Hostname: " $SLURM_JOB_NODELIST
echo "# Job ID: " $SLURM_JOB_ID
echo "# Job Name: " $SLURM_JOB_NAME
echo "# Python file: " $pythonfile
echo "# Options: " $all_opts

# In case of external API usage I saved some API-keys here
if [ -f ~/.api_keys ]; then
    . ~/.api_keys
fi

# For data readin in kgcnn
module load chem/openbabel

# For WandB:
#export WANDB_MODE=offline # no internet connection during calculation on nodes

CONDA_HOME=$(dirname $(dirname $CONDA_EXE))
source $CONDA_HOME/etc/profile.d/conda.sh
conda activate $PYTHON_ENV
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# OpenMP needs this: set stack size to unlimited
ulimit -s unlimited

# Run the Python script with all options
time python3 $pythonfile $all_opts 
#time python3 $pythonfile --gpu 0 --category PAiNN.EnergyForceModel --hyper $config_path
exit_status=$?
if [ $exit_status -ne 0 ]; then
    echo "Error: Python script exited with status $exit_status"
fi
exit $exit_status

