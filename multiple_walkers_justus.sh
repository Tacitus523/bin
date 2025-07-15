#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --time=300:00:00
#SBATCH --output=metadynamics.out
#SBATCH --error=metadynamics.err
#SBATCH --signal=B:USR1@120 # Send the USR1 signal 120 seconds before end of time limit
#SBATCH --gres=gpu:1

N_WALKER=$SLURM_NTASKS_PER_NODE
#GROMACS_PATH="/lustre/home/ka/ka_ipc/ka_he8978/gromacs-nn/bin/GMXRC"
GROMACS_PATH="/lustre/home/ka/ka_ipc/ka_he8978/gromacs-pytorch-cuda/bin/GMXRC"
PYTORCH_ENV="/home/ka/ka_ipc/ka_he8978/miniconda3/envs/pytorch_cuda"

OBSERVATION_SCRIPT=$(which observe_trajectory.py)

export GMX_QMMM_VARIANT=2
export OMP_NUM_THREADS=1
#export GMX_DFTB_ESP=1
#export GMX_DFTB_CHARGES=100
#export GMX_DFTB_QM_COORD=100
#export GMX_DFTB_MM_COORD=100
#export GMX_DFTB_MM_COORD_FULL=100
export GMX_N_MODELS=3 # Number of neural network models
export GMX_MODEL_PATH_PREFIX="model_energy_force" # Prefix of the path to the neural network models
export GMX_ENERGY_PREDICTION_STD_THRESHOLD=0.1  # Threshold for the energy standard deviation between models for a structure to be considered relevant
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
    print_usage
    exit 1
fi
echo "plumed_file: $plumed_file"

# Remaining arguments are additional files
additional_files=("$@")
for file in "${additional_files[@]}"; do
    echo "Additional file: $file"
done

echo "Starting simulation on $SLURM_JOB_NODELIST: $(date)"

source $GROMACS_PATH

module load system/parallel
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

# Handles append identification and append command generation.
# Checks if HILLS files are present, if WALKER_* folders and if .cpt files are present.
generate_append_command() {
    local files=("$@")
    
    local hills_found=false
    for file in "${files[@]}"
    do
        if [ -f "$file" ] && [[ $file == HILLS* ]]
        then
            hills_found=true
            break
        fi
    done

    local walker_found=false
    if $hills_found
    then
        for folder in "${files[@]}"
        do
            if [ -d $folder ] && [[ $folder == WALKER_* ]]
            then
                walker_found=true
                break
            fi
        done
    fi

    local cpt_found=false
    if $walker_found
    then
        for folder in "${files[@]}"
        do
            if [ -d $folder ] && [[ $folder == WALKER_* ]]
            then
                for file in "$folder"/*
                do
                    if [ -f "$file" ] && [[ $file == *".cpt" ]]
                    then
                        cpt_found=true
                        cpt_file=$(basename "$file")
                        append_command="-cpi $cpt_file -append"
                        break 2 # Break out of both loops
                    fi
                done
            fi
        done
    fi

    if $hills_found && ! $walker_found
    then
        echo "HILLS files found, but no WALKER_* folders present. Cannot rerun."
        exit 1
    fi
    if $walker_found && ! $cpt_found
    then
        echo "WALKER_* folders found, but no .cpi files present. Cannot rerun."
        exit 1
    fi

    echo $append_command
}

# Call finalize_job function as soon as we receive USR1 signal
trap 'finalize_job' USR1

basename_tpr=$(basename $tpr_file .tpr)
append_command=$(generate_append_command "${additional_files[@]}")
if [ -n "$append_command" ]
then
    echo "Appending to existing simulation!"
    echo "Append command: $append_command"
fi

sourcedir=$PWD
workdir=$SCRATCH

mkdir -vp $workdir
cp -r -v $tpr_file $plumed_file "${additional_files[@]}" $workdir
cd $workdir

export GMX_MODEL_PATH_PREFIX=$(readlink -f $GMX_MODEL_PATH_PREFIX) # Convert to absolute path

# Create directories for each walker
for i in `seq 0 $((N_WALKER-1))`
do
    mkdir -p WALKER_$i
    cp $plumed_file WALKER_$i
    cp $tpr_file WALKER_$i
    for file in "${additional_files[@]}"
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

# Run each walker in parallel using GNU parallel
run_mdrun() {
    local walker_id=$1
    local tpr_file=$2
    local plumed_file=$3
    local append_command=$4
    local observation_script=$5
    basename_tpr=$(basename $tpr_file .tpr)

    cd "WALKER_$walker_id"
    if [ "$walker_id" -eq 0 ]; then
        gmx mdrun -deffnm $basename_tpr -ntomp 1 -ntmpi 1 -s "$tpr_file" -plumed "$plumed_file" $append_command
    else
        gmx -quiet mdrun -deffnm $basename_tpr -ntomp 1 -ntmpi 1 -s "$tpr_file" -plumed "$plumed_file" $append_command &>> mdrun.out
    fi
    mdrun_pid=$!

    if [ -f "$observation_script" ]
    then
        "$observation_script" --basename $basename_tpr &
        observation_pid=$!
        mexican_standoff $mdrun_pid $observation_pid &
    else
        echo "Observation script not found, running mdrun without observation."
    fi

    echo "Finished walker $walker_id at $(date)"
}

export -f run_mdrun
export OBSERVATION_SCRIPT
export -f mexican_standoff
parallel -j $N_WALKER "run_mdrun {} $tpr_file $plumed_file $append_command \$OBSERVATION_SCRIPT" ::: $(seq 0 $((N_WALKER-1))) &

wait # Wait for all parallel processes to finish, also allows trap finalize_job to be called

cd $sourcedir
rsync -a $workdir/ .
rm -r $workdir
