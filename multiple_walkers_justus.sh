#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --time=300:00:00
#SBATCH --job-name=metadynamics
#SBATCH --output=metadynamics.out
#SBATCH --error=metadynamics.err
#SBATCH --gres=scratch:100 # GB on scratch reserved
#SBATCH --signal=B:USR1@120 # Send the USR1 signal 120 seconds before end of time limit
#SBATCH --gres=gpu:1

# Give the .tpr as $1 and the plumed as $2, and any other files as $3, $4, etc.

N_WALKER=$SLURM_NTASKS_PER_NODE
GROMACS_PATH="/lustre/home/ka/ka_ipc/ka_he8978/gromacs-pytorch-cuda/bin/GMXRC"


echo "Starting multiple walkers: $(date)"

source $GROMACS_PATH

module load system/parallel
module load lib/cudnn/9.0.0_cuda-12.3
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

set -o errexit   # (or set -e) cause batch script to exit immediately when a command fails.
set -o pipefail  # cause batch script to exit immediately also when the command that failed is embedded in a pipeline
set -o nounset   # (or set -u) causes the script to treat unset variables as an error and exit immediately

# Define the signal handler function when job times out
# Note: This is not executed here, but rather when the associated 
# signal is received by the shell.
finalize_job()
{
    # Do whatever cleanup you want here.
    echo "function finalize_job called at $(date)"
    cd $sourcedir
    rsync -a $workdir/ .
    rm -r $workdir
    exit 1
}

# Handles rerun identification and rerun command generation. Checks if HILLS files are present and if WALKER_* folders are present.
generate_rerun_command() {
    local files=$1
    local rerun=false
    local rerun_command=""
    for file in $files
    do
        if [ -f $file ] && [[ $file == HILLS* ]]
        then
            rerun=true
            break
        fi
    done

    if $rerun
    then
        for folder in $files
        do
            if [ -d $folder ] && [[ $folder == WALKER_* ]]
            then
                rerun_command="-cpi state.cpt -append"
                break
            fi
        done
    fi

    if $rerun && [ -z "$rerun_command" ]
    then
        echo "Found HILLS, but no WALKER_* folders, aborting"
        exit 1
    fi

    echo $rerun_command
}

# Call finalize_job function as soon as we receive USR1 signal
trap 'finalize_job' USR1

export GMX_QMMM_VARIANT=2
export OMP_NUM_THREADS=1
#export GMX_DFTB_ESP=1
#export GMX_DFTB_CHARGES=100
#export GMX_DFTB_QM_COORD=100
#export GMX_DFTB_MM_COORD=100
#export GMX_DFTB_MM_COORD_FULL=100
export GMX_N_MODELS=3 # Number of neural network models
export GMX_MODEL_PATH_PREFIX="model_energy_force" # Prefix of the path to the neural network models
export GMX_ENERGY_PREDICTION_STD_THRESHOLD=0.3  # Threshold for the energy standard deviation between models for a structure to be considered relevant
export GMX_FORCE_PREDICTION_STD_THRESHOLD=-1 # Threshold for the force standard deviation between models for a structure to be considered relevant, energy std threshold has priority
export GMX_NN_EVAL_FREQ=10 # Frequency of evaluation of the neural network, 1 means every step
export GMX_NN_ARCHITECTURE="maceqeq" # Architecture of the neural network, hdnnp, schnet or painn for tensorflow, or maceqeq for pytorch
export GMX_NN_SCALER="" # Scaler file for the neural network, optional, empty string if not applicable
export GMX_MAXBACKUP=-1 # Maximum number of backups for the checkpoint file, -1 means none
export PLUMED_MAXBACKUP=-1 # Maximum number of backups for the plumed file, -1 means none

sourcedir=$PWD
workdir=$SCRATCH

tpr_file=$1
shift
plumed_file=$1
shift
other_files=$@

if [ ! -f $tpr_file ]
then
    echo "tpr file not found"
    exit 1
fi
if [[ ! $tpr_file == *.tpr ]]; then
    echo "tpr file must end with .tpr"
    exit 1
fi
if [ ! -f $plumed_file ]
then
    echo "plumed file not found"
    exit 1
fi

mkdir -vp $workdir
cp -r -v $tpr_file $plumed_file $other_files $workdir
cd $workdir

tpr_prefix=$(basename $tpr_file .tpr)
export GMX_MODEL_PATH_PREFIX=$(readlink -f $GMX_MODEL_PATH_PREFIX) # Convert to absolute path

# Create directories for each walker
for i in `seq 0 $((N_WALKER-1))`
do
    mkdir -p WALKER_$i
    cp $plumed_file WALKER_$i
    cp $tpr_file WALKER_$i
    for file in $other_files
    do
        if [ ! -d $file ] && [[ ! $file == HILLS* ]] && [[ ! $file == *.pt ]] # Exclude folders, HILLS files and pytorch model files
        then
            cp $file WALKER_$i
        fi
    done

    cd WALKER_$i/
    sed -i -e 's/NUMWALKER/'${i}'/g' $plumed_file
    cd ..
done

rerun_command=$(generate_rerun_command "$other_files")

# Run each walker in parallel using GNU parallel
run_mdrun() {
    local walker_id=$1
    local tpr_file=$2
    local plumed_file=$3
    local rerun_command=$4
    cd "WALKER_$walker_id"
    if [ "$walker_id" -eq 0 ]; then
        gmx mdrun -ntomp 1 -s "$tpr_file" -plumed "$plumed_file" $rerun_command
    else
        gmx mdrun -ntomp 1 -s "$tpr_file" -plumed "$plumed_file" $rerun_command &>> mdrun.out
    fi
}

export -f run_mdrun
parallel -j $N_WALKER "run_mdrun {} $tpr_file $plumed_file $rerun_command" ::: $(seq 0 $((N_WALKER-1))) &

wait # Wait for all parallel processes to finish, also allows trap finalize_job to be called

cd $sourcedir
rsync -a $workdir/ .
rm -r $workdir
