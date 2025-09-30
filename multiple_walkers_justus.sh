#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --time=300:00:00
#SBATCH --output=metadynamics.out
#SBATCH --error=metadynamics.err
#SBATCH --signal=B:USR1@120 # Send the USR1 signal 120 seconds before end of time limit
##SBATCH --gres=gpu:1

function print_usage() {
    echo "Usage: $0 -t TPR_FILE.tpr [-p PLUMED_FILE.dat] [additional_files...]"
    exit 1
}

#GROMACS_PATH="/lustre/home/ka/ka_ipc/ka_he8978/gromacs-nn/bin/GMXRC"
#GROMACS_PATH="/lustre/home/ka/ka_ipc/ka_he8978/gromacs-pytorch-cuda/bin/GMXRC"
GROMACS_PATH=/lustre/home/ka/ka_ipc/ka_he8978/gromacs-tensorflow/bin/GMXRC
PLUMED_PATH="/lustre/home/ka/ka_ipc/ka_dk5684/sw/plumed-2.5.1-gcc-8.2.0-openblas-release/bin"
PYTORCH_ENV="/home/ka/ka_ipc/ka_he8978/miniconda3/envs/pytorch_cuda"

OBSERVATION_SCRIPT=$(which observe_trajectory.py)

source $GROMACS_PATH
module load system/parallel
module load lib/cudnn/9.0.0_cuda-12.3
export PATH="$PLUMED_PATH:$PATH"
export LD_LIBRARY_PATH="$PYTORCH_ENV/lib:$PYTORCH_ENV/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/lustre/home/ka/ka_ipc/ka_he8978/sw/tensorflow_prebuilt/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/lustre/home/ka/ka_ipc/ka_he8978/sw/jsoncpp_exe/lib64:$LD_LIBRARY_PATH"

export GMX_QMMM_VARIANT=2 # 1 for PME, 2 for cutoff
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
export GMX_NN_ARCHITECTURE="maceqeq" # Architecture of the neural network, hdnnp2nd, hdnnp4th, schnet or painn for tensorflow, or mace, maceqeq, amp for pytorch
export GMX_NN_SCALER="" # Scaler file for the neural network, optional, empty string if not applicable
export GMX_NN_EXTXYZ=2000 # Frequency of writing the extended xyz file, 1 means every step, 0 means never
export GMX_MAXBACKUP=-1 # Maximum number of backups for the checkpoint file, -1 means none
export PLUMED_MAXBACKUP=-1 # Maximum number of backups for the plumed file, -1 means none

# For adding and overwriting environment variables from a file
if [ -f "GMX_VARS.sh" ]; then
    source GMX_VARS.sh
fi

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
    echo "WARNING: Missing plumed file"
    plumed_command=""
else
    if [ ! -f "${plumed_file}" ]; then
        echo "Plumed file ${plumed_file} does not exist" 1>&2
        exit 1
    fi
    echo "plumed_file: $plumed_file"
    plumed_command="-plumed $(basename $plumed_file)"
fi

# Remaining arguments are additional files
additional_files=("$@")
for file in "${additional_files[@]}"
do
    echo "Additional file: $file"
done

if which gmx > /dev/null 2>&1; then
    gmx_command=$(which gmx)
elif which gmx_d > /dev/null 2>&1; then
    gmx_command=$(which gmx_d)
else
    echo "GROMACS command not found. Please check your GROMACS installation."
    exit 1
fi

echo "Starting simulation on $SLURM_JOB_NODELIST: $(date)"

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
    rsync -a $scratch_dir/ .
    rm -r $scratch_dir
    exit 1
}

# Handles append identification and append command generation.
# Checks if HILLS files are present, if WALKER_* folders and if .cpt files are present.
function generate_append_command() {
    local files=("$@")
    
    local append_command=""
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
            if [ -d "$folder" ] && [[ $folder == WALKER_* ]]
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

    echo "$append_command"
}

function observe_trajectory {
    local basename_tpr=$1
    local gmx_pid=$2

    if ! [ -f "$OBSERVATION_SCRIPT" ]
    then
        echo "Observation script $OBSERVATION_SCRIPT not found!" 1>&2
        return 1
    fi

    while true
    do
        sleep 60 # Check every minute
        $OBSERVATION_SCRIPT --basename "$basename_tpr" --once 1> /dev/null
        observation_exit_code=$?
        if [ $observation_exit_code -ne 0 ]
        then
            echo "Observation script detected an explosion. Stopping mdrun."
            kill $gmx_pid 2>/dev/null || true # Stop mdrun if still running
            exit $observation_exit_code
        fi
    done
}

function track_fes_progress {
    while true; do
        sleep 14400 # Check every 4 hours
        date +"%Y-%m-%d %H:%M:%S - Tracking FES progress"
        wc -l HILLS.*
        $HOME/bin/SUM_HILLS
        mkdir -p fes_progress
        mv fes.dat fes_progress/fes_$(date +%Y_%m_%d_%H_%M).dat
        $HOME/bin/plot_scripts/METPES 10 -d fes_progress
    done
}

# Call finalize_job function as soon as we receive USR1 signal
trap 'finalize_job' USR1

sourcedir=$PWD
scratch_dir=$SCRATCH

mkdir -vp $scratch_dir
cp -r $tpr_file $plumed_file "${additional_files[@]}" $scratch_dir
cd $scratch_dir

# Local pathes to copied files
tpr_file=$(basename $tpr_file)
plumed_file=$(basename $plumed_file)
local_additional_files=()
for file in "${additional_files[@]}"
do
    local_additional_files+=("$(basename $file)")
done
additional_files=("${local_additional_files[@]}")
export GMX_MODEL_PATH_PREFIX=$(readlink -f $GMX_MODEL_PATH_PREFIX) # Convert to absolute path in scratch directory

# Append to existing simulation?
append_command=$(generate_append_command "${additional_files[@]}")
if [ -n "$append_command" ]
then
    echo "Appending to existing simulation!"
    echo "Append command: $append_command"
fi

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
    # If appending, add RESTART at the top of the plumed file
    if [ -n "$append_command" ]; then
        tmp_plumed="${plumed_file}.tmp"
        { echo "RESTART"; cat "$plumed_file"; } > "$tmp_plumed"
        mv "$tmp_plumed" "$plumed_file"
    fi
    cd ..
done

# Run each walker in parallel using GNU parallel
function run_mdrun() {
    local walker_id=$1
    shift
    local tpr_file=$1
    shift
    # Remaining arguments
    local mdrun_args="$@"
    basename_tpr=$(basename $tpr_file .tpr)

    cd "WALKER_$walker_id"
    if [ $walker_id -eq 0 ]
    then
        echo "Starting walker $walker_id at $(date)"
        $gmx_command -quiet mdrun -deffnm $basename_tpr -s $tpr_file $mdrun_args &
        run_pid=$!

        observe_trajectory $basename_tpr $run_pid &
        observation_pid=$!
    else
        $gmx_command -quiet mdrun -deffnm $basename_tpr -s $tpr_file $mdrun_args >> mdrun.log 2>&1 &
        run_pid=$!

        observe_trajectory $basename_tpr $run_pid >> mdrun.log 2>&1 &
        observation_pid=$!
    fi

    wait $run_pid
    run_exit_code=$?
    kill $observation_pid 2>/dev/null || true # Stop observation script if still running

    if [ $run_exit_code -ne 0 ]; then
        echo "mdrun for walker $walker_id failed with exit code $run_exit_code"
        exit $run_exit_code
    else
        echo "mdrun for walker $walker_id completed successfully at $(date)"
    fi
}

export -f run_mdrun
export OBSERVATION_SCRIPT
export -f observe_trajectory
export gmx_command
parallel --line-buffer -j $N_WALKER run_mdrun {} $tpr_file $plumed_command $append_command ::: $(seq 0 $((N_WALKER-1))) &
mdrun_pid=$!

track_fes_progress &
track_pid=$!

wait $mdrun_pid # Wait for all parallel processes to finish, also allows trap finalize_job to be called
mdrun_exit_code=$?
disown $track_pid
kill $track_pid 2>/dev/null || true # Stop the FES tracking if still running

cd $sourcedir
rsync -a $scratch_dir/ .
rm -r $scratch_dir

if [ $mdrun_exit_code -eq 0 ]; then
    echo "$(date): All walkers completed successfully."
else
    echo "One or more walkers failed with exit code $mdrun_exit_code"
fi
exit $mdrun_exit_code
