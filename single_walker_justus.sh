#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --time=300:00:00
#SBATCH --output=simulation.out
#SBATCH --error=simulation.err
#SBATCH --signal=B:USR1@120 # Send the USR1 signal 120 seconds before end of time limit
#SBATCH --gres=gpu:1
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=16

#GROMACS_PATH="/lustre/home/ka/ka_ipc/ka_he8978/gromacs-nn/bin/GMXRC"
GROMACS_PATH="/lustre/home/ka/ka_ipc/ka_he8978/gromacs-pytorch-cuda/bin/GMXRC"
# GROMACS_PATH="/lustre/home/ka/ka_ipc/ka_dk5684/sw/gromacs-machine-learning/release/bin/GMXRC"
PYTORCH_ENV="/home/ka/ka_ipc/ka_he8978/miniconda3/envs/pytorch_cuda"

OBSERVATION_SCRIPT=$(which observe_trajectory.py)

# module load chem/orca
# GROMACS_PATH=/lustre/home/ka/ka_ipc/ka_he8978/gromacs-orca/bin/GMXRC
# ORCA_PATH=$(dirname $(which orca))
# export PATH="$ORCA_PATH:$PATH"
# export LD_LIBRARY_PATH="$ORCA_PATH:$LD_LIBRARY_PATH"
# export GMX_ORCA_PATH=$ORCA_PATH

# # Add these after loading the ORCA module
# export OMPI_MCA_rmaps_base_oversubscribe=1
# export NSLOTS=$SLURM_CPUS_PER_TASK
# export HWLOC_HIDE_ERRORS=1

export GMX_QMMM_VARIANT=2
export OMP_NUM_THREADS=1
#export GMX_DFTB_ESP=1
#export GMX_DFTB_CHARGES=100
#export GMX_DFTB_QM_COORD=100
#export GMX_DFTB_MM_COORD=100
#export GMX_DFTB_MM_COORD_FULL=100
export GMX_N_MODELS=1 # Number of neural network models
export GMX_MODEL_PATH_PREFIX="model_energy_force" # Prefix of the path to the neural network models
export GMX_ENERGY_PREDICTION_STD_THRESHOLD=0.05  # Threshold for the energy standard deviation between models for a structure to be considered relevant
export GMX_FORCE_PREDICTION_STD_THRESHOLD=-1 # Threshold for the force standard deviation between models for a structure to be considered relevant, energy std threshold has priority
export GMX_NN_EVAL_FREQ=1 # Frequency of evaluation of the neural network, 1 means every step
export GMX_NN_ARCHITECTURE="maceqeq" # Architecture of the neural network, hdnnp, schnet or painn for tensorflow, or mace,maceqeq, amp for pytorch
export GMX_NN_SCALER="" # Scaler file for the neural network, optional, empty string if not applicable
export GMX_PYTORCH_EXTXYZ=1000 # Frequency of writing the extended xyz file, 1 means every step, 0 means never
export GMX_MAXBACKUP=-1 # Maximum number of backups for the checkpoint file, -1 means none
export PLUMED_MAXBACKUP=-1 # Maximum number of backups for the plumed file, -1 means none

print_usage() {
    echo "Usage: $0 -t TPR_FILE.tpr [-p PLUMED_FILE.dat] [additional_files...]"
    exit 1
}

while getopts ":t:p:" opt; do
    case ${opt} in
        t )
            tpr_file=$OPTARG
            ;;
        p )
            plumed_file=$OPTARG
            ;;
        \? )
            echo "Invalid option: -$OPTARG" 1>&2
            print_usage
            ;;
        : )
            echo "Invalid option: -$OPTARG requires an argument" 1>&2
            print_usage
            ;;
    esac
done
shift $((OPTIND -1))

# Check if mandatory argument is set
if [ -z "${tpr_file}" ]; then
    echo "Missing .tpr file" 1>&2
    print_usage
    exit 1
fi
echo "tpr_file: $tpr_file"

if [ -z "${plumed_file}" ]; then
    echo "Missing plumed file"
    plumed_command=""
else
    echo "plumed_file: $plumed_file"
    plumed_command="-plumed $plumed_file"
fi

# Remaining arguments are additional files
additional_files=("$@")
for file in "${additional_files[@]}"; do
    echo "Additional file: $file"
done

echo "Starting simulation on $SLURM_JOB_NODELIST: $(date)"

source $GROMACS_PATH

module load lib/cudnn/9.0.0_cuda-12.3
export LD_LIBRARY_PATH="$PYTORCH_ENV/lib:$PYTORCH_ENV/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH"

# set -o errexit   # (or set -e) cause batch script to exit immediately when a command fails.
# set -o pipefail  # cause batch script to exit immediately also when the command that failed is embedded in a pipeline
# set -o nounset   # (or set -u) causes the script to treat unset variables as an error and exit immediately

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

# Handles rerun identification and rerun command generation.
# Checks if .xtc/.trr file is present
generate_rerun_command() {
    local files=("$@")
    
    local rerun_command=""
    for file in "${files[@]}"
    do
        if [ -f "$file" ] && [[ $file == *".xtc" || $file == *".trr" ]]
        then
            rerun_command="-rerun $file "
            break
        fi
    done

    echo $rerun_command
}

mexican_standoff() {
    # receive two process IDs. Waits for one of them to finish, then kills the other.
    local pid1=$1 # mdrun process ID
    local pid2=$2 # observation script process ID
    while true
    do
        if ! kill -0 "$pid1" > /dev/null 2>&1
        then
            kill "$pid2" > /dev/null 2>&1
            break
        elif ! kill -0 "$pid2" > /dev/null 2>&1
        then
            # First try SIGTERM
            kill "$pid1" > /dev/null 2>&1
            sleep 60
            # If process still exists, use SIGKILL
            if kill -0 "$pid1" 2>/dev/null
            then
                echo "Process $pid1 did not terminate, sending SIGKILL"
                kill -9 "$pid1" > /dev/null 2>&1
            fi
            break
        fi
        sleep 10 # Sleep for 120 seconds before checking again
    done
}

# Call finalize_job function as soon as we receive USR1 signal
trap 'finalize_job' USR1

rerun_command=$(generate_rerun_command "${additional_files[@]}")
if [ -n "$rerun_command" ]
then
    echo "Reruning existing simulation!"
    echo "Rerun command: $rerun_command"
fi

sourcedir=$PWD
workdir=$SCRATCH

mkdir -vp $workdir
cp -r -v $tpr_file "${additional_files[@]}" $workdir
cd $workdir

export GMX_MODEL_PATH_PREFIX=$(readlink -f $GMX_MODEL_PATH_PREFIX) # Convert to absolute path
basename_tpr=$(basename $tpr_file .tpr)
export GMX_QM_ORCA_BASENAME=$basename_tpr # Set basename for ORCA output files

gmx -quiet mdrun -deffnm $basename_tpr -s "$tpr_file" $rerun_command$plumed_command &
mdrun_pid=$!

if [ -f $OBSERVATION_SCRIPT ]
then
    $OBSERVATION_SCRIPT --basename "$basename_tpr" &
    observation_pid=$!
    mexican_standoff $mdrun_pid $observation_pid &
else
    echo "Observation script not found, running mdrun without observation."
fi

wait # Wait for all parallel processes to finish, also allows trap finalize_job to be called

cd $sourcedir
rsync -a $workdir/ .
rm -r $workdir
