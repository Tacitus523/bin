#!/bin/bash
<<<<<<< HEAD
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=train.out
#SBATCH --error=train.err
#SBATCH --gres=gpu:1

PYTHON_ENV=kgcnn_new

pythonfile=$1
config_path="$2" # optional

# Echo important information into file
echo "# Date: " $(date)
echo "# Hostname: " $SLURM_JOB_NODELIST
echo "# Job ID: " $SLURM_JOB_ID
=======
#$ -l qu=gtx
#$ -q gtx01a,gtx01b,gtx01c,gtx01d,gtx02a,gtx02b,gtx02c,gtx02d,gtx03a,gtx03b,gtx03c,gtx03d,gtx04a,gtx04b,gtx04c,gtx04d,gtx05a,gtx05b,gtx05c,gtx05d,gtx06a,gtx06b,gtx06c,gtx06d
#$ -cwd
#$ -o train.out
#$ -e train.err 

pythonfile=$1
pythonfile=${pythonfile#"/srv/nfs"}

config_path="$2" # optional

# Which GPU?
gpu_id=$( echo $QUEUE | awk '/a/ {print 0} /b/ {print 1}  /c/ {print 2}  /d/ {print 3}')

# How many cores are there?
case $HOSTNAME in
    gtx0[1-6]*)
    cores=10;
    ;;
    gtx0[7-8]*)
    cores=12;
    ;;
    gtx09*)
    cores=16;
    ;;
    gtx10*)
    cores=16;
    ;;
    *)
    echo "Error: Unknown compute node \$HOSTNAME"
    echo "       This script only works for gtx01 thru 10!"
    echo
    exit -1
    ;;
esac

# Echo important information into file
echo "# Hostname: " `hostname`
echo "# Job ID: " $JOB_ID
echo "# gpuid: " $gpu_id
>>>>>>> 9ab32ee2eed1d0a12c9db1b531265e70c5846fdd

# In case of external API usage I saved some API-keys here
if [ -f ~/.api_keys ]; then
    . ~/.api_keys
fi

# For WandB:
export WANDB_MODE=offline # no internet connection during calculation on nodes

<<<<<<< HEAD
CONDA_HOME=$(dirname $(dirname $CONDA_EXE))
source $CONDA_HOME/etc/profile.d/conda.sh
conda activate $PYTHON_ENV

# OpenMP needs this: set stack size to unlimited
ulimit -s unlimited

if [ -z "$config_path" ]
then time python3 $pythonfile -g 0
else time python3 $pythonfile -g 0 -c $config_path
fi

=======
# For data readin in kgcnn
export BABEL_DATADIR="/usr/local/run/openbabel-2.4.1"

export PATH="/home/lpetersen/anaconda_interpreter/bin:$PATH"
source /home/lpetersen/anaconda_interpreter/etc/profile.d/conda.sh
conda activate kgcnn_new

# Deprecated CUDA setting on server
# export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/lib/cuda"
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/run/cuda/lib

# Even older deprecated CUDA setting on server
#CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib

# set OpenMP parallel threads variable:
export OMP_NUM_THREADS=$cores

# OpenMP needs this: set stack size to unlimited
ulimit -s unlimited
# Start time of calculation
start=$( date "+%s" )

if [ -z "$config_path" ]
then time python3 $pythonfile -g $gpu_id
else time python3 $pythonfile -g $gpu_id -c $config_path
fi

# End time of calculation
end=$( date "+%s" )
# Now we calculate the time taken by the calculation
duration=$(( end - start ))
# Now we do Wibbly Wobbly Timey Wimey... Stuff
DAYS=$(( duration / 86400 ))
HOURS=$(( (duration % 86400) / 3600 ))
MINS=$(( ((duration % 86400) % 3600) / 60 ))
SECS=$(( ((duration % 86400) % 3600) % 60 ))
echo "Time taken: $DAYS days, $HOURS hours, $MINS minutes and $SECS seconds."
>>>>>>> 9ab32ee2eed1d0a12c9db1b531265e70c5846fdd
