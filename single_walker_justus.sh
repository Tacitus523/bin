#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --time=300:00:00
#SBATCH --output=simulation.out
#SBATCH --error=simulation.err
#SBATCH --gres=scratch:100 # GB on scratch reserved
#SBATCH --signal=B:USR1@120 # Send the USR1 signal 120 seconds before end of time limit
#SBATCH --gres=gpu:1

#GROMACS_PATH="/lustre/home/ka/ka_ipc/ka_he8978/gromacs-nn/bin/GMXRC"
GROMACS_PATH="/lustre/home/ka/ka_ipc/ka_he8978/gromacs-pytorch-cuda/bin/GMXRC"
#GROMACS_PATH="/lustre/home/ka/ka_ipc/ka_dk5684/sw/gromacs-machine-learning/release/bin/GMXRC"
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
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

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
export GMX_N_MODELS=3 # Number of neural network models
export GMX_MODEL_PATH_PREFIX="model_energy_force" # Prefix of the path to the neural network models
export GMX_ENERGY_PREDICTION_STD_THRESHOLD=0.05  # Threshold for the energy standard deviation between models for a structure to be considered relevant
export GMX_FORCE_PREDICTION_STD_THRESHOLD=-1 # Threshold for the force standard deviation between models for a structure to be considered relevant, energy std threshold has priority
export GMX_NN_EVAL_FREQ=10 # Frequency of evaluation of the neural network, 1 means every step
export GMX_NN_ARCHITECTURE="maceqeq" # Architecture of the neural network, hdnnp, schnet or painn for tensorflow, or maceqeq for pytorch
export GMX_NN_SCALER="" # Scaler file for the neural network, optional, empty string if not applicable
export GMX_MAXBACKUP=-1 # Maximum number of backups for the checkpoint file, -1 means none
export PLUMED_MAXBACKUP=-1 # Maximum number of backups for the plumed file, -1 means none

sourcedir=$PWD
workdir=$SCRATCH

mkdir -vp $workdir
cp -r -v $tpr_file "${additional_files[@]}" $workdir
if [ -n "$plumed_command" ]; then
    cp -v $plumed_file $workdir
fi
cd $workdir

export GMX_MODEL_PATH_PREFIX=$(readlink -f $GMX_MODEL_PATH_PREFIX) # Convert to absolute path
basename_tpr=$(basename $tpr_file .tpr)

gmx mdrun -deffnm $basename_tpr -ntomp 1 -ntmpi 1 -s "$tpr_file" $plumed_command &

wait # Wait for all parallel processes to finish, also allows trap finalize_job to be called

cd $sourcedir
rsync -a $workdir/ .
rm -r $workdir
