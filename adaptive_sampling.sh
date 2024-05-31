# For use with an adaptive sampling capable dftb-gromacs build

GROMACS_NN_PATH="/lustre/home/ka/ka_ipc/ka_he8978/gromacs-nn/bin/GMXRC" # Path to compiled gromacs bin with installed GMXRC
BLAS_PATH="/lustre/home/ka/ka_ipc/ka_dk5684/sw/OpenBLAS-0.3.10-release/lib"
PLUMED_PATH="/lustre/home/ka/ka_ipc/ka_dk5684/sw/plumed-2.5.1-gcc-8.2.0-openblas-release/lib"
#GROMACS_NN_PATH="/data/lpetersen/gromacs-adaptive-sampling/bin/GMXRC" # Path to compiled gromacs bin with installed GMXRC
#BLAS_PATH="/usr/local/run/OpenBLAS-0.3.10/lib"
#PLUMED_PATH="/usr/local/run/plumed-2.5.1-openblas/lib"
GMX_N_TF_MODELS=3 # Amount of trained models to use for the adaptive sampling, suffix of the models
GMX_TF_MODEL_PATH_PREFIX="model_energy_force" # prefix of the models
GMX_ENERGY_PREDICTION_STD_THRESHOLD=0.002 # Threshold for the energy standard deviation between models for a structure to be considered relevant
GMX_FORCE_PREDICTION_STD_THRESHOLD=0.004 # Threshold for the force standard deviation between models for a structure to be considered relevant, energy std threshold has priority

N_ITERATIONS=10 # Amount of repeated adaptive samplings+retrainings
INITIAL_ITERATION=2 # Used to restart after a previous adaptive sampling, unique actions for initialization will not be performed if > 0 (such as DELETING existing content)
N_SAMPLES=1000 # Amount of new structures to be added to the dataset
BATCH_SIZE=50 # Amount of samples per independent parallel sampling runs

AS_MDP_FILE="run.mdp" # mdp file for adaptive sampling, TODO: cat extra line for charge?
PLUMED_FILE="plumed.dat" # plumed file for adaptive sampling, optional, empty string if not applicable
# TODO:Index file per sampling

START_GEOMETRIES_PREFIX="/lustre/work/ws/ws1/ka_he8978-thiol_disulfide/training_data/B3LYP_aug-cc-pVTZ_water/start_geometries/GEOM_" # Prefix(+Path) of folders with start geometries and topology, suffix should be continous integers starting from 0
START_GEOMETRIES_SUFFIX_PADDED=true # If the suffix of the folders is padded with zeros to make them appear in order, true or false
START_GEOMETRIES_GRO_NAME="geom_sol.gro" # file name of the .gro in the folders above
START_GEOMETRIES_TOPOL_NAME="topol.top" # file name of the .top in the folders above
# TODO:Index file per molecule

GROMACS_ORCA_PATH="/lustre/home/ka/ka_ipc/ka_he8978/gromacs-orca/bin/GMXRC"
ORCA_MODULE="chem/orca/5.0.4" # Orca module on Justus2
ORCA_FOLDER_PREFIX="orca_job_" # Just used as the name to generate the folders with orca calculations 
ORCA_BASENAME="sp" # all .top., .mdp, .gro .ORCAINFO, .ndx should start with this as well
#.ORCAINFO can contain a 1-liner %PAL section for parallelisation
# TODO: Index file and extra line for charge in mdp?

# For Orca result extraction and dataset expansion
PYTHON_ENV="kgcnn_new"
AT_COUNT=15 # Amount of atoms in QM system. Could get it in theory from wc -l sp.inp and wc -l sp.ORCAINFO, but I don't have a way to to the forces for different system sizes yet anyway.
GEOM_FILE="/lustre/work/ws/ws1/ka_he8978-thiol_disulfide/training_data/B3LYP_aug-cc-pVTZ_water/ThiolDisulfidExchange.xyz" # Unit: Angstrom
ENERGY_FILE="/lustre/work/ws/ws1/ka_he8978-thiol_disulfide/training_data/B3LYP_aug-cc-pVTZ_water/energy_diff.txt" # Unit: Hartree
ENERGY_REFERENCE_FILE="/lustre/work/ws/ws1/ka_he8978-thiol_disulfide/training_data/B3LYP_aug-cc-pVTZ_water/energy_opt.txt" # to calculate the difference to a reference energy, contains energies of optimized fragments, optional, Unit: Hartree
CHARGE_FILE="/lustre/work/ws/ws1/ka_he8978-thiol_disulfide/training_data/B3LYP_aug-cc-pVTZ_water/charges.txt" # Unit: e
FORCE_FILE="/lustre/work/ws/ws1/ka_he8978-thiol_disulfide/training_data/B3LYP_aug-cc-pVTZ_water/forces.xyz" # Unit: Hartree/Bohr
ESP_FILE="/lustre/work/ws/ws1/ka_he8978-thiol_disulfide/training_data/B3LYP_aug-cc-pVTZ_water/esps_by_mm.txt" # Unit: V
ESP_GRADIENT_FILE="/lustre/work/ws/ws1/ka_he8978-thiol_disulfide/training_data/B3LYP_aug-cc-pVTZ_water/esp_gradients.txt" # Unit: e/Bohr^2
DATA_SOURCES_FILE="/lustre/work/ws/ws1/ka_he8978-thiol_disulfide/training_data/B3LYP_aug-cc-pVTZ_water/data_sources.txt" # Optional, Intended to keep track of different sampling methods. 1D column of strings with sampling description for plotting
CUTOFF=10 # Cutoff for bond existence, should be larger than the symm func cutoff, in Angstrom
TOTAL_CHARGE=-1 # Charge of the molecule, no support for different charges right now
DATASET_PREFIX="ThiolDisulfidExchange"
TRAINING_INDICES_FILE="training_indices.pkl"

BABEL_PATH="/opt/bwhpc/common/chem/openbabel/3.1.1"

# For Retraining
EPOCHS=200 # Epochs each model gets trained with the new dataset

DEBUG=false # Prints all gromacs outputs, if true. Otherwise redirects to /dev/null

##### Util ######
remove_folder_if_exists() {
    local folder_to_delete=$1
    if [ -d "$folder_to_delete" ]
    then rm -r "$folder_to_delete"
    fi
}

extraction_script="/lustre/home/ka/ka_ipc/ka_he8978/bin/extract_qmmm_information.sh"
energy_diff_script="/lustre/home/ka/ka_ipc/ka_he8978/bin/calculate_energy_diff.py"
preparation_script="/lustre/home/ka/ka_ipc/ka_he8978/kgcnn_fork/prepare_data.py"
retraining_script="/lustre/home/ka/ka_ipc/ka_he8978/kgcnn_fork/transfer_learning.py"

######################################################################
######################START UNIQUE ACTIONS############################
######################################################################
if [ -z "$LD_LIBARY_PATH" ]
then LD_LIBRARY_PATH=""
fi

set -o errexit   # (or set -e) cause batch script to exit immediately when a command fails.
set -o pipefail  # cause batch script to exit immediately also when the command that failed is embedded in a pipeline
set -o nounset   # (or set -u) causes the script to treat unset variables as an error and exit immediately 

CONDA_HOME=$(dirname $(dirname $CONDA_EXE))
source $CONDA_HOME/etc/profile.d/conda.sh
conda activate $PYTHON_ENV

absolute_model_prefix="$(readlink -f "$GMX_TF_MODEL_PATH_PREFIX")"
absolute_mdp_file_path="$(readlink -f "$AS_MDP_FILE")"
absolute_plumed_file_path="$(readlink -f "$PLUMED_FILE")"
absolute_training_indices_file_path="$(readlink -f "$TRAINING_INDICES_FILE")"

out_file="adaptive_sampling.out"
error_file="adaptive_sampling.err"
out_file="$(readlink -f "$out_file")"
error_file="$(readlink -f "$error_file")"

if grep -iq "nprocs" $ORCA_BASENAME.ORCAINFO
then
  n_parallel_tasks=`grep -i nprocs $ORCA_BASENAME.ORCAINFO | awk '{print $3}'` # Intented for 1-liner %PAL section, fails otherwise
else
  n_parallel_tasks=1
fi
echo "Using $n_parallel_tasks cores per Orca calculation"

if [ $INITIAL_ITERATION -le 0 ]
then
    if [ -f $out_file ]
    then rm $out_file
    fi

    if [ -f $error_file ]
    then rm $error_file
    fi
fi

cat << EOM >> $out_file
MODEL_PREFIX: $absolute_model_prefix
AS_MDP_FILE: $absolute_mdp_file_path
PLUMED_FILE: $absolute_plumed_file_path
ENERGY STD_THRESHOLD: $GMX_ENERGY_PREDICTION_STD_THRESHOLD
FORCE STD_THRESHOLD: $GMX_FORCE_PREDICTION_STD_THRESHOLD
N_ITERATIONS: $N_ITERATIONS
N_SAMPLES: $N_SAMPLES
BATCH_SIZE: $BATCH_SIZE
EPOCHS: $EPOCHS

START_GEOMETRIES_PREFIX: $START_GEOMETRIES_PREFIX
START_GEOMETRIES_GRO_NAME: $START_GEOMETRIES_GRO_NAME
START_DATASET_FOLDER: $(dirname $GEOM_FILE)
EOM

source_dir=$PWD
root_dir="adaptive_sampling"
current_training_data_folder="current_training_data"
mkdir -p "$root_dir"
cd "$root_dir"
if [ $INITIAL_ITERATION -le 0 ]
then
    remove_folder_if_exists "sampling"
    remove_folder_if_exists "orca_calculations"
    remove_folder_if_exists "retraining"
    remove_folder_if_exists "$current_training_data_folder"
    mkdir -p "sampling"
    mkdir -p "orca_calculations"
    mkdir -p "orca_calculations/failed_calculations"
    mkdir -p "retraining"
    mkdir -p "$current_training_data_folder"
fi

start_geometries_dirname=$(dirname $START_GEOMETRIES_PREFIX)
start_geometries_prefix=$(basename $START_GEOMETRIES_PREFIX)
n_start_geometries=$(find $start_geometries_dirname -maxdepth 1 -type d -name "$start_geometries_prefix*" | wc -l)
if [ "$n_start_geometries" -le 0 ]
then
    echo "No starting geometries were found in folders starting with the prefix $start_geometries_prefix in $start_geometries_dirname"
    exit 1
fi
padding_length_iterations=${#N_ITERATIONS}
padding_length_start_geometries=${#n_start_geometries}
padding_length_samples=${#N_SAMPLES}

root_dir=$(readlink -f ".")
current_training_data_folder="$(readlink -f "$current_training_data_folder")"
cd "$current_training_data_folder"
if [ $INITIAL_ITERATION -le 0 ]
then
    cp $GEOM_FILE .
    cp $ENERGY_FILE .
    cp $CHARGE_FILE .
    cp $FORCE_FILE .
    cp $ESP_FILE .
    cp $ESP_GRADIENT_FILE .
fi
current_geom_file="$(readlink -f "$(basename $GEOM_FILE)")"
current_energy_file="$(readlink -f "$(basename $ENERGY_FILE)")"
current_charge_file="$(readlink -f "$(basename $CHARGE_FILE)")"
current_force_file="$(readlink -f "$(basename $FORCE_FILE)")"
current_esp_file="$(readlink -f "$(basename $ESP_FILE)")"
current_esp_gradient_file="$(readlink -f "$(basename $ESP_GRADIENT_FILE)")"
if [ -n "$DATA_SOURCES_FILE" ]
then 
    if [ $INITIAL_ITERATION -le 0 ]
    then cp $DATA_SOURCES_FILE .
    fi
    current_data_source_file="$(readlink -f "$(basename $DATA_SOURCES_FILE)")"
else
    DATA_SOURCES_FILE="data_sources.txt"
    current_data_source_file="$(readlink -f $DATA_SOURCES_FILE)"
    dataset_size=$(cat $ENERGY_FILE | wc -l) # Could also be done with awk, but just wc -l also prints the file name
    for data_point_idx in $(seq 1 $dataset_size)
    do 
        echo "Original Dataset" >> "$current_data_source_file"
    done
fi

cat << EOM > data_prep_config.json
{
  "DATA_FOLDER": "$current_training_data_folder",
  "GEOMETRY_FILE": "$(basename $current_geom_file)",
  "ENERGY_FILE": "$(basename $current_energy_file)",
  "CHARGE_FILE": "$(basename $current_charge_file)",
  "ESP_FILE": "$(basename $current_esp_file)",
  "ESP_GRAD_FILE": "$(basename $current_esp_gradient_file)",
  "AT_COUNT": $AT_COUNT,
  "CUTOFF": $CUTOFF,
  "FORCE_FILE": "$(basename $current_force_file)",
  "TOTAL_CHARGE": $TOTAL_CHARGE,
  "PREFIX": "$DATASET_PREFIX",
  "TARGET_FOLDER": "$current_training_data_folder"
}
EOM

cd "$root_dir"

cp $absolute_training_indices_file_path "retraining/training_indices.pkl"
cat << EOM > retraining/retraining_config.json
{
  "DATA_DIRECTORY": "$current_training_data_folder",
  "DATASET_NAME": "$DATASET_PREFIX",
  "MODEL_PREFIX": "$absolute_model_prefix",
  "EPOCHS": $EPOCHS
}
EOM

if $DEBUG
then redirect=""
else 
    redirect='> /dev/null 2>&1' # Redirects gromacs outputs to nowhere
    export TF_CPP_MIN_LOG_LEVEL=2 # Supresses some Tensorflow warnings/infos
fi
######################################################################
########################END UNIQUE ACTIONS############################
######################################################################

sampling_dependency="" # Initialize the training loop without dependency for the sampling
for ((iteration_idx=$INITIAL_ITERATION; iteration_idx<$N_ITERATIONS; iteration_idx++))
do
iteration_idx_padded="$(printf "%0${padding_length_iterations}d" "$iteration_idx")"
#######################################################################
########################START ADAPTIVE SAMPLING########################
#######################################################################
# TODO: Index file per sampling

sampling_job_ids="" # Initialize the orca job ids as emptpy array
# Iterate over lower limits of batch sizes
for batch_idx in $(seq 0 $BATCH_SIZE $((N_SAMPLES-1)))
do
    lower_limit=$batch_idx
    upper_limit=$((lower_limit+BATCH_SIZE-1))
    if [ $upper_limit -gt $N_SAMPLES ]
    then
        upper_limit=$N_SAMPLES
    fi

    batch_idx=$(printf "%0${padding_length_samples}d" $batch_idx) # Pads with zeros

    batch_name_samp=samp_${iteration_idx_padded}_$batch_idx
    sampler_prefix=run #_$batch_idx # Used to name the files unique for each batch
    sampler_folder=$batch_name_samp

    cd "sampling"
    mkdir -p "$sampler_folder"
    cd "$sampler_folder"

    if [ "$lower_limit" = 0 ]
    then 
        sampler_out_file="$out_file"
        sampler_error_file="$error_file"
    else 
        sampler_out_file="sampling.out"
        sampler_error_file="sampling.err"
    fi

    sampling_jobfile="sampling_jobfile.sge"
    cat << EOM > "$sampling_jobfile"
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --job-name=$batch_name_samp
#SBATCH --output=$sampler_out_file
#SBATCH --error=$sampler_error_file
#SBATCH --open-mode=append
#SBATCH --kill-on-invalid-dep=yes
$sampling_dependency
#SBATCH --signal=B:USR1@120 # Send the USR1 signal 120 seconds before end of time limit

echo "JOB: \$SLURM_JOB_ID started on \$SLURM_JOB_NODELIST -- \$(date)"

# Define the signal handler function when job times out
# Note: This is not executed here, but rather when the associated 
# signal is received by the shell.
finalize_job()
{
    # Do whatever cleanup you want here.
    echo "function finalize_job called at \$(date)"
    exit 0
}

# Call finalize_job function as soon as we receive USR1 signal
trap 'finalize_job' USR1

# #$ -N $batch_name_samp
# #$ -cwd
# #$ -o $sampler_out_file
# #$ -e $sampler_error_file
# #$ -l qu=gtx
# #$ -hold_jid "retrain_$((iteration_idx - 1))"

# Function to pad a number with zeros
pad_with_zeros() {
    local num=\$1
    local target_length=\$2
    printf "%0\${target_length}d" "\$num"
}

source "$GROMACS_NN_PATH"
export GMX_MAXBACKUP=-1
export PLUMED_MAXBACKUP=-1
export GMX_QMMM_VARIANT=2
export OMP_NUM_THREADS=1
export GMX_N_TF_MODELS=$GMX_N_TF_MODELS # Amount of trained models to use for the adaptive sampling, suffix of the models
export GMX_TF_MODEL_PATH_PREFIX=$absolute_model_prefix # prefix of the models
export GMX_FORCE_PREDICTION_STD_THRESHOLD=$GMX_FORCE_PREDICTION_STD_THRESHOLD # Threshold for the deviation between models for a structure to be considered relevant

export LD_LIBRARY_PATH=$BLAS_PATH:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$PLUMED_PATH:$LD_LIBRARY_PATH
ulimit -s unlimited

# Clean extraction results of previous run
if [ -f $root_dir/orca_calculations/geoms.xyz ]
then
    rm $root_dir/orca_calculations/geoms.xyz
fi
if [ -f $root_dir/orca_calculations/charges_hirsh.txt ]
then
    rm $root_dir/orca_calculations/charges_hirsh.txt
fi
if [ -f $root_dir/orca_calculations/forces.xyz ]
then
    rm $root_dir/orca_calculations/forces.xyz
fi
if [ -f $root_dir/orca_calculations/energies.txt ]
then
    rm $root_dir/orca_calculations/energies.txt
fi
if [ -f $root_dir/orca_calculations/energy_diff.txt ]
then
    rm $root_dir/orca_calculations/energy_diff.txt
fi
if [ -f $root_dir/orca_calculations/esps_by_mm.txt ]
then
    rm $root_dir/orca_calculations/esps_by_mm.txt
fi
if [ -f $root_dir/orca_calculations/esp_gradients.txt ]
then
    rm $root_dir/orca_calculations/esp_gradients.txt
fi

echo "ITERATION $iteration_idx_padded"
for sampler_job_idx in \$(seq $lower_limit $upper_limit)
do
    sampler_job_idx=\$(pad_with_zeros \$sampler_job_idx $padding_length_samples) # padded
    # copy a random start structure in the folder
    random_folder_number=\$((5112 + \$RANDOM % ($n_start_geometries-5112)))
    if $START_GEOMETRIES_SUFFIX_PADDED
    then
        random_folder_number=\$(pad_with_zeros "\$random_folder_number" "$padding_length_start_geometries")
    fi
    random_folder=$START_GEOMETRIES_PREFIX\$random_folder_number
    random_geometry=\$random_folder/$START_GEOMETRIES_GRO_NAME
    random_topol=\$random_folder/$START_GEOMETRIES_TOPOL_NAME
    echo \$random_folder_number >> "starting_structure_idxs.txt"
    # TODO:Index file per molecule

    orca_sampler_folder="$root_dir/orca_calculations/$ORCA_FOLDER_PREFIX\$sampler_job_idx"
    # Clean previous run
    if [ -d "\$orca_sampler_folder" ]
        then rm -r "\$orca_sampler_folder"
    fi
    if [ -f $sampler_prefix.gro ]
    then
        rm $sampler_prefix.gro
    fi
    if [ -f $sampler_prefix.top ]
    then
        rm $sampler_prefix.top
    fi

    cp \$random_geometry $sampler_prefix.gro
    cp \$random_topol $sampler_prefix.top

    gmx grompp -f $absolute_mdp_file_path -c $sampler_prefix.gro -p $sampler_prefix.top -maxwarn 1 -o $sampler_prefix.tpr $redirect # TODO: Index file
    if [ -n "$absolute_plumed_file_path" ]
    then
        gmx mdrun -ntomp 1 -ntmpi 1 -deffnm $sampler_prefix -plumed $absolute_plumed_file_path $redirect &
    else
        gmx mdrun -ntomp 1 -ntmpi 1 -deffnm $sampler_prefix $redirect &
    fi
    wait # wait for gmx to finish while still allowing the trap to be executed.

    if [ -f adaptive_sampling.xtc ]
    then
        echo 0 | gmx trjconv -s $sampler_prefix.tpr -f adaptive_sampling.xtc -o $ORCA_BASENAME.gro $redirect # Single entry .xtc, has to be converted directly because the .tpr might change
        echo "Sampling step \$sampler_job_idx: \$(grep step $ORCA_BASENAME.gro)" # For output in .out
        cat $ORCA_BASENAME.gro >> "adaptive_sampling.gro"
        
        mkdir -p "\$orca_sampler_folder"
        mv adaptive_sampling.xtc \$orca_sampler_folder/$ORCA_BASENAME.xtc
        mv $ORCA_BASENAME.gro \$orca_sampler_folder/$ORCA_BASENAME.gro
        cp $sampler_prefix.top \$orca_sampler_folder/topol.top
    fi
done
EOM
    sampling_job_id=$(sbatch $sampling_jobfile | awk '{print $4}')
    if [ -z "$sampling_job_ids" ]
    then sampling_job_ids="$sampling_job_id"
    else sampling_job_ids="$sampling_job_ids:$sampling_job_id"
    fi

#######################################################################
#########################END ADAPTIVE SAMPLING#########################
#######################################################################

#######################################################################
########################START ORCA CALCULATIONS########################
#######################################################################

    cd "$root_dir/orca_calculations"

    batch_name_orca=orca_${iteration_idx_padded}_$batch_idx

    if [ "$lower_limit" = 0 ]
    then 
        batch_out_file="$out_file"
        batch_error_file="$error_file"
    else 
        batch_out_file="$batch_name_orca.out"
        batch_error_file="$batch_name_orca.err"
    fi

    orca_jobfile="orca_jobfile_$batch_idx.sge" 
    cat << EOM > $orca_jobfile
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=$n_parallel_tasks
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=6:00:00
#SBATCH --job-name=$batch_name_orca
#SBATCH --output=$batch_out_file
#SBATCH --error=$batch_error_file
#SBATCH --open-mode=append
#SBATCH --dependency=afterok:$sampling_job_id
#SBATCH --kill-on-invalid-dep=yes
#SBATCH --gres=scratch:100 # GB on scratch reserved

# #$ -N $batch_name_orca
# #$ -cwd
# #$ -o $batch_out_file
# #$ -e $batch_error_file
# #$ -l qu=gtx
# #$ -hold_jid "samp_${iteration_idx}_$batch_idx"

module load $ORCA_MODULE
orca_command=\`which orca\`
orca_path=\$(dirname \$orca_command)

echo "JOB: \$SLURM_JOB_ID started on \$SLURM_JOB_NODELIST -- \$(date)"
echo "Workdir: \$SCRATCH"

source "$GROMACS_ORCA_PATH"
export PATH=\$orca_path:\$PATH
export LD_LIBRARY_PATH=$BLAS_PATH:\$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$PLUMED_PATH:\$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=\$orca_path:\$LD_LIBRARY_PATH
export GMX_ORCA_PATH=\$orca_path
export GMX_MAXBACKUP=-1
export GMX_QMMM_VARIANT=2
export OMP_NUM_THREADS=1
export GMX_QM_ORCA_BASENAME=$ORCA_BASENAME
export HWLOC_HIDE_ERRORS=2 # I have no idea, hwloc sometimes throws errors, that mess up orca. This seems to suppress them, which makes the orca calculations work

ulimit -s unlimited

for suffix in \$(seq $lower_limit $upper_limit)
do
    echo "JOB: Geometry $lower_limit..\$suffix..$upper_limit -- \$(date)"
    padded_suffix=\$(printf "%0${padding_length_samples}d" \$suffix)
    folder="${ORCA_FOLDER_PREFIX}\${padded_suffix}"

    if [ ! -d "\$folder" ]
    then 
        continue
    fi

    cd \$folder

    current_dir=\$PWD
    workdir="\$SCRATCH/orca_\$suffix"
    mkdir -p \$workdir

    cp topol.top \$workdir
    cp $ORCA_BASENAME.gro \$workdir
    cp $source_dir/$ORCA_BASENAME.mdp \$workdir
    cp $source_dir/$ORCA_BASENAME.ORCAINFO \$workdir
    # TODO: Index file
      
    cd \$workdir  

    gmx grompp -f $ORCA_BASENAME.mdp -c $ORCA_BASENAME.gro -p topol.top -o $ORCA_BASENAME.tpr -maxwarn 1 $redirect
    gmx mdrun -ntomp 1 -ntmpi 1 -deffnm $ORCA_BASENAME -rerun $redirect

    rm $ORCA_BASENAME.densities
    rm $ORCA_BASENAME.gbw

    cd \$current_dir
    rsync -a \$workdir/ .
    rm -rf \$workdir

    cd ..
done

EOM
    
    cd "$root_dir"
done # batches of sampling and orca
echo "Submitted samplings in iteration $iteration_idx_padded with job id/s $sampling_job_ids"

# Iterate again to submit orca jobs after sampling jobs, potential erros with changed names in the upper part
cd orca_calculations
orca_job_ids="" # Initialize the orca job ids as emptpy array
for batch_idx in $(seq 0 $BATCH_SIZE $((N_SAMPLES-1)))
do
    batch_idx=$(printf "%0${padding_length_samples}d" $batch_idx) # Pads with zeros
    orca_jobfile="orca_jobfile_$batch_idx.sge" 
    orca_job_id=$(sbatch $orca_jobfile | awk '{print $4}')
    if [ -z "$orca_job_ids" ]
    then orca_job_ids="$orca_job_id"
    else orca_job_ids="$orca_job_ids:$orca_job_id"
    fi
done
echo "Submitted orca calculations in iteration $iteration_idx_padded with job id/s $orca_job_ids"
cd "$root_dir"

#######################################################################
#########################END ORCA CALCULATIONS#########################
#######################################################################

#######################################################################
############################START RETRAINING###########################
#######################################################################
cd "retraining"
iteration_folder="iteration_$iteration_idx_padded"
mkdir -p $iteration_folder
cd $iteration_folder

retraining_jobfile="retraining_jobfile.sge"
cat << EOM > $retraining_jobfile
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=retrain_${iteration_idx_padded}
#SBATCH --output=$out_file
#SBATCH --error=$error_file
#SBATCH --open-mode=append
#SBATCH --dependency=afterok:$orca_job_ids
#SBATCH --kill-on-invalid-dep=yes
#SBATCH --gres=gpu:1

# #$ -N retrain_${iteration_idx}
# #$ -cwd
# #$ -o $out_file
# #$ -e $error_file
# #$ -l qu=gtx
# #$ -hold_jid "orca_${iteration_idx}_*"

set -o errexit   # (or set -e) cause batch script to exit immediately when a command fails.
set -o pipefail  # cause batch script to exit immediately also when the command that failed is embedded in a pipeline
# set -o nounset   # (or set -u) causes the script to treat unset variables as an error and exit immediately 

echo "JOB: \$SLURM_JOB_ID started on \$SLURM_JOB_NODELIST -- \$(date)"

######## SOME MERGING OF PARALLEL DATA COLLECTION ########
cd "$root_dir/sampling"

if ls samp_${iteration_idx_padded}_*/adaptive_sampling.gro > /dev/null 2>&1 # Check if there are any adaptive_sampling.gro in the sampling folders
then
    cat samp_${iteration_idx_padded}_*/adaptive_sampling.gro >> "adaptive_sampling_$iteration_idx_padded.gro"
    for file in samp_${iteration_idx_padded}_*/adaptive_sampling.gro # Prevent double appending
    do
        mv \$file \${file%/*}/adaptive_sampling_${iteration_idx_padded}.gro
    done
fi

cd "$root_dir/orca_calculations"

# OpenMP needs this: set stack size to unlimited
ulimit -s unlimited

if [ ! -f geoms.xyz ] # Prevent double appending depending on the existence of the geometries file. orca_calculations gets cleaned every sampling restart
then
    # Check for orca calculation failures, rename folder on failure
    for job_folder in $ORCA_FOLDER_PREFIX*
    do
        if ! grep -q "FINAL SINGLE" \$job_folder/$ORCA_BASENAME.out 2> /dev/null
        then
            echo "\$job_folder failed"
            if [ -f "\$job_folder/$ORCA_BASENAME.out" ]
            then
                mv "\$job_folder" "failed_calculations/\${job_folder}_${iteration_idx_padded}" # keep .out file for error message
            fi
            rm -r \$job_folder
        else
            echo "Adaptive Sampling $iteration_idx_padded" >> "$current_data_source_file"
        fi
    done

    # Extract orca calculations results
    $extraction_script $AT_COUNT $ORCA_FOLDER_PREFIX $ORCA_BASENAME
    cat geoms.xyz >> $current_geom_file
    cat charges_hirsh.txt >> $current_charge_file
    cat forces.xyz >> $current_force_file
    cat esps_by_mm.txt >> $current_esp_file
    cat esp_gradients.txt >> $current_esp_gradient_file
    cat_exit_status=\$?
    if [ ! \$cat_exit_status -eq 0 ]; then
        echo "ESP calculation failed"
        exit 1
    fi

    if [ -n "$ENERGY_REFERENCE_FILE" ]
    then
        python3 $energy_diff_script -e energies.txt -r $ENERGY_REFERENCE_FILE -o energy_diff.txt
        cat energy_diff.txt >> $current_energy_file
    else
        echo "No reference energy supplied, appending raw energies!"
        cat energies.txt >> $current_energy_file
    fi
fi

################# DATA MERGING DONE ####################
export BABEL_DATADIR=$BABEL_PATH
python3 $preparation_script -c ${current_training_data_folder}/data_prep_config.json 

cd "$root_dir/retraining/$iteration_folder"
cp "../training_indices.pkl" .
python3 $retraining_script -g 0 -c ../retraining_config.json
cp "training_indices.pkl" ..

EOM

retraining_job_id=$(sbatch $retraining_jobfile | awk '{print $4}')
sampling_dependency="#SBATCH --dependency=afterok:$retraining_job_id"
echo "Submitted retraining in iteration $iteration_idx_padded with job id $retraining_job_id"
cd "$root_dir"
#######################################################################
#############################END RETRAINING############################
#######################################################################
done # iterations

cd ..