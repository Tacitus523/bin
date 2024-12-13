#!/bin/bash
#$ -cwd

# Give the .tpr as $1 and the plumed as $2, and any other files as $3, $4, etc.

n_walker=$NSLOTS # Number of walkers to run, from SGE environment variable NSLOTS
GROMACS_PATH="/home/tkubar/GMX-DFTB/gromacs-dftbplus/release-machine-learning/bin/GMXRC"

echo "Starting multiple walkers: $(date)"
echo "Running on $(hostname)"
echo "Job ID: $JOB_ID"
echo "Number of slots: $NSLOTS"

source $GROMACS_PATH

set -o errexit   # (or set -e) cause batch script to exit immediately when a command fails.
set -o pipefail  # cause batch script to exit immediately also when the command that failed is embedded in a pipeline
set -o nounset   # (or set -u) causes the script to treat unset variables as an error and exit immediately


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
                rerun_command="-append"
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
export GMX_QMMM_VARIANT=1
export OMP_NUM_THREADS=1
#export GMX_DFTB_ESP=1
#export GMX_DFTB_CHARGES=100
#export GMX_DFTB_QM_COORD=100
#export GMX_DFTB_MM_COORD=100
# #export GMX_DFTB_MM_COORD_FULL=100
# export GMX_N_TF_MODELS=1
# export GMX_ENERGY_PREDICTION_STD_THRESHOLD=-1
# export GMX_FORCE_PREDICTION_STD_THRESHOLD=-1
export GMX_MAXBACKUP=-1
export PLUMED_MAXBACKUP=-1

sourcedir=$PWD
workdir=/scratch/$USER/metadynamic_$JOB_ID

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
# export GMX_TF_MODEL_PATH_PREFIX=$(readlink -f "model_energy_force") # Just for MLMM

# Create directories for each walker
for i in `seq 0 $((n_walker-1))`
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

# Run each walker
jobname=$(basename $tpr_file .tpr) # Use the tpr file name as the job name
for i in `seq 0 $((n_walker-1))`
do
cd WALKER_$i

gmx_d mdrun -ntomp 1 -deffnm $jobname -plumed $plumed_file $rerun_command &> mdrun.out &
cd ..
done

wait

cd $sourcedir
rsync -a $workdir/ .
rm -r $workdir
