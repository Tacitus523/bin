#!/bin/bash
#$ -cwd

function print_usage() {
    echo "Usage: $0 -t TPR_FILE.tpr [-p PLUMED_FILE.dat] [additional_files...]"
    exit 1
}

if [ "$SGE_O_HOST" == "hydrogen2" ]
then
    N_WALKER=$NSLOTS # Number of walkers to run, from SGE environment variable NSLOTS
    GROMACS_PATH="/usr/local/run/gromacs-dftbplus-machine-learning/bin/GMXRC"
    PLUMED_PATH="/usr/local/run/plumed-2.9.0/bin" # Just for fes progress tracking
elif [ "$SGE_O_HOST" == "tcbserver2" ]
then
    N_WALKER=$N_WALKER # Number of walkers to run, from SGE environment variable NSLOTS doesnt work on TCB
    #GROMACS_PATH="/home/tkubar/GMX-DFTB/gromacs-dftbplus/release-machine-learning/bin/GMXRC"
    #GROMACS_PATH="/data/lpetersen/gromacs-pytorch-cuda-float64/bin/GMXRC"
    GROMACS_PATH="/home/lpetersen/sw/gromacs_dftb_pytorch/bin/GMXRC"
    #GROMACS_PATH="/home/lpetersen/sw/gromacs_dftb/bin/GMXRC"
    PLUMED_PATH="/usr/local/run/plumed-2.8.3/bin" # Just for fes progress tracking
    PYTORCH_ENV="/home/conda/envs/pytorch_cuda_2.3.1"

    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib/gcc/x86_64-linux-gnu/10"
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/run/OpenBLAS-0.3.10/lib"
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/run/plumed-2.5.1-openblas/lib"
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PYTORCH_ENV/lib/python3.12/site-packages/torch/lib"
else
    echo "Unknown host: $SGE_O_HOST. Cannot set environment."
    exit 1
fi
PARALLEL_PATH="/home/lpetersen/sw/parallel_exe/bin"

source $GROMACS_PATH
export PATH=$PARALLEL_PATH:$PLUMED_PATH:$PATH

# Default environment variables for GROMACS with DFTB and machine learning, can be overwritten inside GMX_VARS.sh
export GMX_QMMM_VARIANT=2 # 1 for PME, 2 for cutoff
export OMP_NUM_THREADS=1
#export GMX_DFTB_ESP=1
#export GMX_DFTB_CHARGES=100
#export GMX_DFTB_QM_COORD=100
#export GMX_DFTB_MM_COORD=100
#export GMX_DFTB_MM_COORD_FULL=100
export GMX_N_MODELS=1 # Number of neural network models
export GMX_MODEL_PATH_PREFIX="model_energy_force" # Prefix of the path to the neural network models
export GMX_ENERGY_PREDICTION_STD_THRESHOLD=0.1  # Threshold for the energy standard deviation between models for a structure to be considered relevant
export GMX_FORCE_PREDICTION_STD_THRESHOLD=-1 # Threshold for the force standard deviation between models for a structure to be considered relevant, energy std threshold has priority
export GMX_NN_EVAL_FREQ=1 # Frequency of evaluation of the neural network, 1 means every step
export GMX_NN_ARCHITECTURE="maceqeq" # Architecture of the neural network, hdnnp2nd, hdnnp4th, schnet or painn for tensorflow, or mace, maceqeq, amp for pytorch
export GMX_NN_SCALER="" # Scaler file for the neural network, optional, empty string if not applicable
export GMX_PYTORCH_EXTXYZ=1000 # Frequency of writing the extended xyz file, 1 means every step, 0 means never
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
    plumed_command="-plumed $plumed_file"
fi

# Remaining arguments are additional files
additional_files=("$@")
for file in "${additional_files[@]}"
do
    echo "Additional file: $file"
done

if [ -z "$JOB_ID" ]
then
    JOB_ID=$(date +%s) # Use current timestamp as job ID if not set
    echo "JOB_ID not set, using current timestamp: $JOB_ID"
fi

if [ -z "$N_WALKER" ]
then
    N_WALKER=1 # Default to 1 walker if not set
    echo "N_WALKER not set, defaulting to 1"
fi

if which gmx > /dev/null 2>&1; then
    gmx_command=$(which gmx)
elif which gmx_d > /dev/null 2>&1; then
    gmx_command=$(which gmx_d)
else
    echo "GROMACS command not found. Please check your GROMACS installation."
    exit 1
fi
sourcedir=$PWD
scratchdir_prefix="/scratch"
scratch_dir_suffix="${USER}/metadynamic_$JOB_ID"
scratch_dir="$scratchdir_prefix/$scratch_dir_suffix"
scratch_dir_on_login="$scratchdir_prefix/${HOSTNAME%.ipc.kit.edu}/$scratch_dir_suffix" # For printing on login node

echo "Starting multiple walkers: $(date)"
echo "Running on $HOSTNAME"
echo "Scratch directory: $scratch_dir_on_login"
echo "Job ID: $JOB_ID"
echo "Number of walkers: $N_WALKER"

set -o errexit   # (or set -e) cause batch script to exit immediately when a command fails.
set -o pipefail  # cause batch script to exit immediately also when the command that failed is embedded in a pipeline
set -o nounset   # (or set -u) causes the script to treat unset variables as an error and exit immediately

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

function track_fes_progress {
    while true; do
        sleep 14400 # Check every 4 hours
        date +"%Y-%m-%d %H:%M:%S - Tracking FES progress"
        wc -l HILLS.*
        /home/lpetersen/bin/SUM_HILLS
        mkdir -p fes_progress
        mv fes.dat fes_progress/fes_$(date +%Y_%m_%d_%H_%M).dat
        /home/lpetersen/bin/plot_scripts/METPES 10 -d fes_progress
    done
}

mkdir -vp $scratch_dir
cp -r $tpr_file $plumed_file "${additional_files[@]}" $scratch_dir
cd $scratch_dir

export GMX_MODEL_PATH_PREFIX=$(readlink -f $GMX_MODEL_PATH_PREFIX) # Convert to absolute path

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

    if [ $walker_id -eq 0 ]
    then
        echo "Starting walker $walker_id at $(date)"
        $gmx_command -quiet mdrun -deffnm $basename_tpr -s $tpr_file $mdrun_args
        return_code=$?
    else
        $gmx_command -quiet mdrun -deffnm $basename_tpr -s $tpr_file $mdrun_args >> mdrun.log 2>&1
        return_code=$?
    fi

    if [ $return_code -ne 0 ]; then
        echo "Walker $walker_id failed with exit code $return_code at $(date)"
    else
        echo "Walker $walker_id completed successfully at $(date)"
    fi
    return $return_code
}
# export gmx_command
# export -f run_mdrun
# export parent_pid=$$
# parallel --line-buffer -j $N_WALKER run_mdrun {} $tpr_file $plumed_command $append_command ::: $(seq 0 $((N_WALKER-1))) &
# parallel_pid=$!

# for i in `seq 0 $((N_WALKER-1))`
# do
#     cd WALKER_$i/
#     run_mdrun $i $tpr_file $plumed_command $append_command &
#     cd ..
# done

function start_runs() {
    local tpr_file=$1
    shift
    local mdrun_args="$@"

    pids=()
    for i in `seq 0 $((N_WALKER-1))`
    do
        cd WALKER_$i/
        run_mdrun $i $tpr_file $mdrun_args &
        pids+=($!)
        cd ..
    done

    wait

    exit_code=0
    for pid in "${pids[@]}"; do
        wait $pid
        pid_exit_code=$?
        if [ $pid_exit_code -ne 0 ]; then
            exit_code=$pid_exit_code
        fi
    done
    return $exit_code
}

start_runs $tpr_file $plumed_command $append_command &
mdrun_pid=$!

track_fes_progress &
track_pid=$!

#wait $parallel_pid # Wait for all walkers to finish, allowing trap to handle cleanup on failure
wait $mdrun_pid # Wait for all walkers to finish, allowing trap to handle cleanup on failure
mdrun_exit_code=$?
kill $track_pid 2>/dev/null || true # Stop the FES tracking if still running

cd $sourcedir
rsync -a $scratch_dir/ .
rsync_exit_code=$?

if [ $rsync_exit_code -ne 0 ]; then
    echo "Rsync failed with exit code $rsync_exit_code"
    echo "Results remain in $scratch_dir"
else
    rm -r $scratch_dir
fi

if [ $mdrun_exit_code -eq 0 ]; then
    echo "$(date): All walkers completed successfully."
else
    echo "One or more walkers failed with exit code"
fi
exit $mdrun_exit_code
