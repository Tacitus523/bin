#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --time=300:00:00
#SBATCH --output=simulation.out
#SBATCH --error=simulation.err
#SBATCH --signal=B:USR1@120 # Send the USR1 signal 120 seconds before end of time limit

print_usage() {
    echo "Usage: $0 -t <tpr_file> [-p <plumed_file>] [-s <simulation_type>] [additional_files...]"
    exit 1
}

simulation_type="pytorch"
while getopts ":t:p:s:" opt; do
    case ${opt} in
        t )
            tpr_file=$OPTARG
            ;;
        p )
            plumed_file=$OPTARG
            ;;
        s)
            simulation_type=$OPTARG
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

# General environment setup
export GMX_QMMM_VARIANT=2 # Cutoff for QM/MM interactions
export OMP_NUM_THREADS=1
export GMX_MAXBACKUP=-1 # Maximum number of backups for the checkpoint file, -1 means none
export PLUMED_MAXBACKUP=-1 # Maximum number of backups for the plumed file, -1 means none

case "$simulation_type" in
    "pytorch"*|"tensorflow"*)
        if [[ $simulation_type == "pytorch" ]]
        then
            echo "Using PyTorch simulation type"
            GROMACS_PATH="/lustre/home/ka/ka_ipc/ka_he8978/gromacs-pytorch-cuda/bin/GMXRC"
            PYTORCH_ENV="/home/ka/ka_ipc/ka_he8978/miniconda3/envs/pytorch_cuda"
        elif [[ $simulation_type == "pytorch_cpu" ]]
        then
            echo "Using PyTorch CPU simulation type"
            GROMACS_PATH="/lustre/home/ka/ka_ipc/ka_he8978/gromacs-pytorch-cpu/bin/GMXRC"
            PYTORCH_ENV="/home/ka/ka_ipc/ka_he8978/miniconda3/envs/pytorch_cpu"
        elif [[ $simulation_type == "tensorflow" ]]
        then
            echo "Using TensorFlow simulation type"
            GROMACS_PATH="/home/ka/ka_ipc/ka_he8978/gromacs-tensorflow/bin/GMXRC"
            export LD_LIBRARY_PATH="/lustre/home/ka/ka_ipc/ka_he8978/sw/tensorflow_prebuilt/lib:$LD_LIBRARY_PATH"
            export LD_LIBRARY_PATH="/lustre/home/ka/ka_ipc/ka_he8978/sw/jsoncpp_exe/lib64:$LD_LIBRARY_PATH"
        else
            echo "Unknown PyTorch simulation type: $simulation_type"
            exit 1
        fi
        parallel_flag="-ntomp 1 -ntmpi 1"
        module load lib/cudnn/9.0.0_cuda-12.3
        export LD_LIBRARY_PATH="$PYTORCH_ENV/lib:$PYTORCH_ENV/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH"
        export GMX_N_MODELS=3 # Number of neural network models
        export GMX_MODEL_PATH_PREFIX="model_energy_force" # Prefix of the path to the neural network models
        export GMX_ENERGY_PREDICTION_STD_THRESHOLD=0.5  # Threshold for the energy standard deviation between models for a structure to be considered relevant
        export GMX_FORCE_PREDICTION_STD_THRESHOLD=-1 # Threshold for the force standard deviation between models for a structure to be considered relevant, energy std threshold has priority
        export GMX_NN_EVAL_FREQ=100 # Frequency of evaluation of the neural network, 1 means every step
        export GMX_NN_ARCHITECTURE="maceqeq" # Architecture of the neural network, hdnnp2nd, hdnnp4th, schnet or painn for tensorflow, or mace, maceqeq, amp for pytorch
        export GMX_NN_SCALER="" # Scaler file for the neural network, optional, empty string if not applicable
        export GMX_NN_EXTXYZ=1000 # Frequency of writing the extended xyz file, 1 means every step, 0 means never

        #OBSERVATION_SCRIPT=$(which observe_trajectory.py) # Used to recognize explosions in the simulation
        ;;
    "orca")
        export HWLOC_HIDE_ERRORS=1 # Hide errors from hwloc, which is used by ORCA
        module load chem/orca
        GROMACS_PATH="/lustre/home/ka/ka_ipc/ka_he8978/gromacs-orca/bin/GMXRC"
        parallel_flag="-ntomp 1 -ntmpi 1" 
        ORCA_PATH=$(dirname $(which orca))
        export PATH="$ORCA_PATH:$PATH"
        export LD_LIBRARY_PATH="$ORCA_PATH:$LD_LIBRARY_PATH"
        export GMX_ORCA_PATH=$ORCA_PATH
        # basename set in submit_single_walker_justus.sh
        ;;
    "dftb")
        GROMACS_PATH="/lustre/home/ka/ka_ipc/ka_dk5684/sw/gromacs-machine-learning/release/bin/GMXRC"
        parallel_flag=""
        #export GMX_DFTB_ESP=1
        #export GMX_DFTB_CHARGES=100
        #export GMX_DFTB_QM_COORD=100
        #export GMX_DFTB_MM_COORD=100
        #export GMX_DFTB_MM_COORD_FULL=100
        ;;
    *)
        echo "Unknown simulation type: $simulation_type"
        exit 1
esac

source $GROMACS_PATH
if which gmx > /dev/null 2>&1; then
    gmx_command=$(which gmx)
elif which gmx_d > /dev/null 2>&1; then
    gmx_command=$(which gmx_d)
else
    echo "GROMACS command not found. Please check your GROMACS installation."
    exit 1
fi

# For adding and overwriting environment variables from a file
if [ -f "GMX_VARS.sh" ]; then
    source GMX_VARS.sh
fi

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
cp -r -v $tpr_file $plumed_file "${additional_files[@]}" $workdir
cd $workdir

basename_tpr=$(basename $tpr_file .tpr)

$gmx_command -quiet mdrun $parallel_flag -deffnm $basename_tpr $rerun_command $plumed_command &
mdrun_pid=$!

if [ -f "$OBSERVATION_SCRIPT" ]
then
    $OBSERVATION_SCRIPT --basename "$basename_tpr" &
    observation_pid=$!
    mexican_standoff $mdrun_pid $observation_pid &
else
    echo "Observation script not found, running mdrun without observation."
fi

wait $mdrun_pid
simulation_exit_code=$?
# Wait for all parallel processes to finish, also allows trap finalize_job to be called

cd $sourcedir
rsync -a $workdir/ .
rm -r $workdir

if [ $simulation_exit_code -ne 0 ]; then
    echo "$(date): Simulation failed with exit code $simulation_exit_code"
    exit $simulation_exit_code
fi
echo "$(date): Simulation finished successfully."
