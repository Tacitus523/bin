#!/bin/bash
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --mem=12G
#SBATCH --time=48:00:00
#SBATCH --job-name=metadynamics
#SBATCH --output=metadynamics.out
#SBATCH --error=metadynamics.err
#SBATCH --gres=scratch:100 # GB on scratch reserved
#SBATCH --signal=B:USR1@120 # Send the USR1 signal 120 seconds before end of time limit

# Give the .tpr as $1 and the plumed as $2, and any other files as $3, $4, etc.
N_WALKER=16
GROMACS_PATH="/lustre/home/ka/ka_ipc/ka_he8978/gromacs-nn/bin/GMXRC"

echo "Starting multiple walkers: $(date)"

source $GROMACS_PATH
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

# Call finalize_job function as soon as we receive USR1 signal
trap 'finalize_job' USR1

export GMX_QMMM_VARIANT=2
export OMP_NUM_THREADS=1
#export GMX_DFTB_ESP=1
#export GMX_DFTB_CHARGES=100
#export GMX_DFTB_QM_COORD=100
#export GMX_DFTB_MM_COORD=100
#export GMX_DFTB_MM_COORD_FULL=100
export GMX_N_TF_MODELS=1
export GMX_ENERGY_PREDICTION_STD_THRESHOLD=-1
export GMX_FORCE_PREDICTION_STD_THRESHOLD=-1
export GMX_MAXBACKUP=-1
export PLUMED_MAXBACKUP=-1

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
if [ ! -f $plumed_file ]
then
    echo "plumed file not found"
    exit 1
fi

mkdir -vp $workdir
cp -r -v $tpr_file $plumed_file $other_files $workdir
cd $workdir

export GMX_TF_MODEL_PATH_PREFIX=$(readlink -f "model_energy_force") # Just for MLMM

# Create directories for each walker
for i in `seq 0 $((N_WALKER-1))`
do
mkdir WALKER_$i
cp $plumed_file WALKER_$i
cp $tpr_file WALKER_$i
for file in $other_files
do
    if [ -f $file ]
    then
        cp $file WALKER_$i
    fi
done

cd WALKER_$i/
sed -i -e 's/NUMWALKER/'${i}'/g' $plumed_file
cd ..
done

# Run each walker
for i in `seq 0 $((N_WALKER-1))`
do
cd $workdir/WALKER_$i

gmx mdrun -ntomp 1 -s $tpr_file -plumed $plumed_file &> mdrun.out &
cd ..
done

wait

cd $sourcedir
rsync -a $workdir/ .
rm -r $workdir
