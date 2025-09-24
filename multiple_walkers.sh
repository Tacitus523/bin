#!/bin/bash
#$ -cwd
#$ -l qu=gtx
#$ -q gtx01a,gtx01b,gtx01c,gtx01d,gtx02a,gtx02b,gtx02c,gtx02d,gtx03a,gtx03b,gtx03c,gtx03d,gtx05a,gtx05b,gtx05c,gtx05d,gtx06a,gtx06b,gtx06c,gtx06d,gtx09a,gtx09b,gtx09c,gtx09d,gtx10a,gtx10b,gtx10c,gtx10d

# Give the .tpr as $1 and the plumed as $2, and any other files as $3, $4, etc.

N_WALKER=$NSLOTS # Number of walkers to run, from SGE environment variable NSLOTS
GROMACS_PATH="/usr/local/run/gromacs-dftbplus-machine-learning/bin/GMXRC"
PLUMED_PATH="/usr/local/run/plumed-2.9.0/bin" # Just for fes progress tracking

# N_WALKER=$N_WALKER # Number of walkers to run, from SGE environment variable NSLOTS doesnt work on TCB
# #GROMACS_PATH="/home/tkubar/GMX-DFTB/gromacs-dftbplus/release-machine-learning/bin/GMXRC"
# #GROMACS_PATH="/data/lpetersen/gromacs-pytorch-cuda-float64/bin/GMXRC"
# GROMACS_PATH="/home/lpetersen/sw/gromacs_dftb_pytorch/bin/GMXRC"
# #GROMACS_PATH="/home/lpetersen/sw/gromacs_dftb/bin/GMXRC"
# PYTORCH_ENV="/home/conda/envs/pytorch_cuda_2.3.1"

PARALLEL_PATH="/home/lpetersen/sw/parallel_exe/bin"

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
export GMX_NN_ARCHITECTURE="maceqeq" # Architecture of the neural network, hdnnp, schnet or painn for tensorflow, or mace,maceqeq, amp for pytorch
export GMX_NN_SCALER="" # Scaler file for the neural network, optional, empty string if not applicable
export GMX_PYTORCH_EXTXYZ=1000 # Frequency of writing the extended xyz file, 1 means every step, 0 means never
export GMX_MAXBACKUP=-1 # Maximum number of backups for the checkpoint file, -1 means none
export PLUMED_MAXBACKUP=-1 # Maximum number of backups for the plumed file, -1 means none


source $GROMACS_PATH
export PATH=$PARALLEL_PATH:$PLUMED_PATH:$PATH
# export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib/gcc/x86_64-linux-gnu/10"
# export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/run/OpenBLAS-0.3.10/lib"
# export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/run/plumed-2.5.1-openblas/lib"
# export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PYTORCH_ENV/lib/python3.12/site-packages/torch/lib"

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

echo "Starting multiple walkers: $(date)"
echo "Running on $(hostname)"
echo "Job ID: $JOB_ID"
echo "Number of walkers: $N_WALKER"

set -o errexit   # (or set -e) cause batch script to exit immediately when a command fails.
set -o pipefail  # cause batch script to exit immediately also when the command that failed is embedded in a pipeline
set -o nounset   # (or set -u) causes the script to treat unset variables as an error and exit immediately

# Cleanup function for graceful exit
cleanup() {
    local exit_code=$?
    echo "Cleaning up on exit (code: $exit_code)"
    if [ -n "${workdir:-}" ] && [ -d "$workdir" ]; then
        echo "Copying results back from $workdir"
        cd "$sourcedir" 2>/dev/null || true
        rsync -a "$workdir/" . 2>/dev/null || true
        rm -rf "$workdir" 2>/dev/null || true
    fi
    exit $exit_code
}

# Set trap for cleanup
trap cleanup EXIT INT TERM


# Handles append identification and append command generation.
# Checks if HILLS files are present, if WALKER_* folders and if .cpt files are present.
generate_append_command() {
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

    echo $append_command
}

function track_fes_progress {
    while true; do
        sleep 21600 # Every 6 hours
        date +"%Y-%m-%d %H:%M:%S - Tracking FES progress"
        wc -l HILLS.*
        /home/lpetersen/bin/SUM_HILLS
        mkdir -p fes_progress
        mv fes.dat fes_progress/fes_$(date +%Y_%m_%d_%H_%M).dat
        /home/lpetersen/bin/plot_scripts/METPES 10 -d fes_progress
    done
}

sourcedir=$PWD
workdir=/scratch/$USER/metadynamic_$JOB_ID

mkdir -vp $workdir
cp -r -v $tpr_file $plumed_file "${additional_files[@]}" $workdir
cd $workdir

tpr_prefix=$(basename $tpr_file .tpr)
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

append_command=$(generate_append_command "$additional_files")

# Run each walker in parallel using GNU parallel
run_mdrun() {
    local walker_id=$1
    local tpr_file=$2
    local plumed_command=$3
    local append_command=$4
    basename_tpr=$(basename $tpr_file .tpr)

    cd "WALKER_$walker_id"
    if [ "$walker_id" -eq 0 ]; then
        gmx -quiet mdrun -deffnm $basename_tpr -s "$tpr_file" $plumed_command $append_command
    else
        gmx -quiet mdrun -deffnm $basename_tpr -s "$tpr_file" $plumed_command $append_command &>> mdrun.out
    fi
    echo "Finished walker $walker_id at $(date)"
}

echo $LD_LIBRARY_PATH
export -f run_mdrun
parallel -j $N_WALKER "run_mdrun {} $tpr_file $plumed_command $append_command" ::: $(seq 0 $((N_WALKER-1))) &
run_pid=$!

track_fes_progress &
track_pid=$!

wait $run_pid # Wait for all walkers to finish, allowing trap to handle cleanup on failure
run_exit_code=$?
kill $track_pid || true # Stop the FES tracking if still running

cd $sourcedir
rsync -a $workdir/ .
rsync_exit_code=$?

if [ $rsync_exit_code -ne 0 ]; then
    echo "Rsync failed with exit code $rsync_exit_code"
    echo "Results remain in $workdir"
else
    rm -r $workdir
fi

if [ $run_exit_code -eq 0 ]; then
    echo "All walkers completed successfully."
else
    echo "One or more walkers failed with exit code $run_exit_code"
fi
exit $run_exit_code
