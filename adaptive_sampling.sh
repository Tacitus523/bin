# For use with an adaptive sampling capable dftb-gromacs build

GROMACS_NN_PATH="/data/lpetersen/gromacs-adaptive-sampling/bin/GMXRC" # Path to compiled gromacs bin with installed GMXRC
BLAS_PATH="/usr/local/run/OpenBLAS-0.3.10/lib"
PLUMED_PATH="/usr/local/run/plumed-2.5.1-openblas/lib"
GMX_N_TF_MODELS=3 # Amount of trained models to use for the adaptive sampling, suffix of the models
GMX_TF_MODEL_PATH_PREFIX="model_energy_force" # prefix of the models
GMX_FORCE_PREDICTION_STD_THRESHOLD=0.005 # Threshold for the deviation between models for a structure to be considered relevant

ITERATIONS=10 # Amount of repeated adaptive samplings+retrainings
N_SAMPLES=1000 # Amount of new structures to be added to the dataset
BATCH_SIZE=100 # Amount of samples per independent parallel sampling runs

AS_MDP_FILE="run.mdp" # mdp file for adaptive sampling, TODO: cat extra line for charge?
PLUMED_FILE="plumed.dat" # plumed file for adaptive sampling, optional, empty string if not applicable
# TODO:Index file per sampling

START_GEOMETRIES_PREFIX="/data/lpetersen/training_data/B3LYP_aug-cc-pVTZ_water/start_geometries/GEOM_" # Prefix(+Path) of folders with start geometries and topology, suffix should be continous integers starting from 0
START_GEOMETRIES_SUFFIX_PADDED=true # If the suffix of the folders is padded with zeros to make them appear in order, true or false
START_GEOMETRIES_GRO_NAME="geom_sol.gro" # file name of the .gro in the folders above
START_GEOMETRIES_TOPOL_NAME="topol.top" # file name of the .top in the folders above
# TODO:Index file per molecule

GROMACS_ORCA_PATH="/data/lpetersen/gromacs-orca/bin/GMXRC"
ORCA_PATH="/usr/local/run/orca_5_0_4_linux_x86-64_shared_openmpi411"
ORCA_FOLDER_PREFIX="orca_job_"
ORCA_BASENAME="sp" # all .top., .mdp, .gro .ORCAINFO, .ndx should start with this as well
# TODO: Index file and extra line for charge in mdp?

# For Orca result extraction and dataset expansion
AT_COUNT=15 # Amount of atoms in QM system. Could get it in theory from wc -l sp.inp and wc -l sp.ORCAINFO, but I don't have a way to to the forces for different system sizes yet anyway.
GEOM_FILE="/data/lpetersen/training_data/B3LYP_aug-cc-pVTZ_water/ThiolDisulfidExchange.xyz"
ENERGY_FILE="/data/lpetersen/training_data/B3LYP_aug-cc-pVTZ_water/energy_diff.txt"
ENERGY_REFERENCE_FILE="/data/lpetersen/training_data/B3LYP_aug-cc-pVTZ_water/energy_opt.txt" # to calculate the difference to a reference energy, contains energies of optimized fragments, optional
CHARGE_FILE="/data/lpetersen/training_data/B3LYP_aug-cc-pVTZ_water/charges.txt"
FORCE_FILE="/data/lpetersen/training_data/B3LYP_aug-cc-pVTZ_water/forces.xyz"
ESP_FILE="/data/lpetersen/training_data/B3LYP_aug-cc-pVTZ_water/esps_by_mm.txt"
ESP_GRADIENT_FILE="/data/lpetersen/training_data/B3LYP_aug-cc-pVTZ_water/esp_gradients.txt"
DATA_SOURCES_FILE="/data/lpetersen/training_data/B3LYP_aug-cc-pVTZ_water/data_sources.txt" # Optional, Intended to keep track of different sampling methods. 1D column of strings with sampling description for plotting
CUTOFF=10 # Cutoff for bond existence, should be larger than the symm func cutoff, in Angstrom
TOTAL_CHARGE=-1 # Charge of the molecule, no support for different charges right now
DATASET_PREFIX=ThiolDisulfidExchange

# For Retraining
EPOCHS=100

DEBUG=false # Prints all gromacs outputs, if true. Otherwise redirects to /dev/null

##### Util ######
remove_folder_if_exists() {
    local folder_to_delete=\$1
    if [ -f "$folder_to_delete" ]
    then rm -r "$folder_to_delete/*"
    fi
}

######################################################################
######################START UNIQUE ACTIONS############################
######################################################################
absolute_model_prefix="$(readlink -f "$GMX_TF_MODEL_PATH_PREFIX")"
absolute_mdp_file_path="$(readlink -f "$AS_MDP_FILE")"
absolute_plumed_file_path="$(readlink -f "$PLUMED_FILE")"

out_file="adaptive_sampling.out"
error_file="adaptive_sampling.err"
out_file="$(readlink -f "$out_file")"
error_file="$(readlink -f "$error_file")"

if [ -f $out_file ]
then rm $out_file
fi

if [ -f $error_file ]
then rm $error_file
fi

cat << EOM > $out_file
MODEL_PREFIX: $absolute_model_prefix
AS_MDP_FILE: $absolute_mdp_file_path
PLUMED_FILE: $absolute_plumed_file_path
STD_THRESHOLD: $GMX_FORCE_PREDICTION_STD_THRESHOLD
ITERATIONS: $ITERATIONS
N_SAMPLES: $N_SAMPLES
BATCH_SIZE: $BATCH_SIZE
EPOCHS: $EPOCHS

START_GEOMETRIES_PREFIX: $START_GEOMETRIES_PREFIX
START_GEOMETRIES_GRO_NAME: $START_GEOMETRIES_GRO_NAME
START_DATASET_FOLDER: $(dirname $GEOM_FILE)
EOM

source_dir=$PWD
current_training_data_folder="current_training_data"
mkdir -p "adaptive_sampling"
cd "adaptive_sampling"
remove_folder_if_exists "sampling"
remove_folder_if_exists "orca_calculations"
remove_folder_if_exists "retraining"
remove_folder_if_exists "$current_training_data_folder"
mkdir -p "sampling"
mkdir -p "orca_calculations"
mkdir -p "retraining"
mkdir -p "$current_training_data_folder"

start_geometries_dirname=$(dirname $START_GEOMETRIES_PREFIX)
start_geometries_prefix=$(basename $START_GEOMETRIES_PREFIX)
n_start_geometries=$(find $start_geometries_dirname -maxdepth 1 -type d -name "$start_geometries_prefix*" | wc -l)
if [ "$n_start_geometries" -le 0 ]
then
    echo "No starting geometries were found in folders starting with the prefix $start_geometries_prefix in $start_geometries_dirname"
    exit 1
fi
padding_length_start_geometries=${#n_start_geometries}
padding_length_samples=${#N_SAMPLES}

current_training_data_folder="$(readlink -f "$current_training_data_folder")"
cd "$current_training_data_folder"
cp $GEOM_FILE .
cp $ENERGY_FILE .
cp $CHARGE_FILE .
cp $FORCE_FILE .
cp $ESP_FILE .
cp $ESP_GRADIENT_FILE .
current_geom_file="$(readlink -f "$(basename $GEOM_FILE)")"
current_energy_file="$(readlink -f "$(basename $ENERGY_FILE)")"
current_charge_file="$(readlink -f "$(basename $CHARGE_FILE)")"
current_force_file="$(readlink -f "$(basename $FORCE_FILE)")"
current_esp_file="$(readlink -f "$(basename $ESP_FILE)")"
current_esp_gradient_file="$(readlink -f "$(basename $ESP_GRADIENT_FILE)")"
if [ -n "$DATA_SOURCES_FILE" ]
then 
    cp $DATA_SOURCES_FILE .
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
  "AT_COUNT": "$AT_COUNT",
  "CUTOFF": "$CUTOFF",
  "FORCE_FILE": "$(basename $current_force_file)",
  "TOTAL_CHARGE": "$TOTAL_CHARGE",
  "PREFIX": "$DATASET_PREFIX",
  "TARGET_FOLDER": "$current_training_data_folder"
}
EOM

cd ..

cat << EOM > retraining/retraining_config.json
{
  "DATA_DIRECTORY": "$current_training_data_folder",
  "DATASET_NAME": "$DATASET_PREFIX",
  "MODEL_PREFIX": "$absolute_model_prefix",
  "EPOCHS": "$EPOCHS"
}
EOM

if $DEBUG
then redirect=""
else redirect='> /dev/null 2>&1'
fi
######################################################################
########################END UNIQUE ACTIONS############################
######################################################################

for ((iteration_idx=0; iteration_idx<$ITERATIONS; iteration_idx++))
do
#######################################################################
########################START ADAPTIVE SAMPLING########################
#######################################################################
# TODO: Index file per sampling

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

    batch_name_samp=samp_${iteration_idx}_$batch_idx
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
#$ -N $batch_name_samp
#$ -cwd
#$ -o $sampler_out_file
#$ -e $sampler_error_file
#$ -l qu=gtx
#$ -hold_jid "retrain_$((iteration_idx - 1))"

# Function to pad a number with zeros
pad_with_zeros() {
    local num=\$1
    local target_length=\$2
    printf "%0\${target_length}d" "\$num"
}

# Which GPU?
gpu_id=\`   echo \$QUEUE | awk '/a/ {print 0} /b/ {print 1}  /c/ {print 2}  /d/ {print 3}'\`

# How many cores are there?
case \$HOSTNAME in
    gtx0[1-6]*)
    cores=10;
    ;;
    gtx0[7-8]*)
    cores=12;
    ;;
    gtx09* | gtx10*)
    cores=16;
    ;;
    *)
    echo "Error: Unknown compute node \$HOSTNAME"
    echo "       This script only works for gtx01 thru 10!"
    echo
    exit -1
    ;;
esac

# Offset for pinning on cores
pinoffset=\`echo \$QUEUE \$cores | awk '/a/ {print 0} /b/ {print \$2} /c/ {print 2*\$2} /d/ {print 3*\$2}'\`

source "$GROMACS_NN_PATH"
export GMX_MAXBACKUP=-1
export PLUMED_MAXBACKUP=-1
export GMX_QMMM_VARIANT=2
export OMP_NUM_THREADS=1
export GMX_N_TF_MODELS=$GMX_N_TF_MODELS # Amount of trained models to use for the adaptive sampling, suffix of the models
export GMX_TF_MODEL_PATH_PREFIX=$absolute_model_prefix # prefix of the models
export GMX_FORCE_PREDICTION_STD_THRESHOLD=$GMX_FORCE_PREDICTION_STD_THRESHOLD # Threshold for the deviation between models for a structure to be considered relevant

BLAS_PATH="$BLAS_PATH"
PLUMED_PATH="$PLUMED_PATH"

export LD_LIBRARY_PATH=\$BLAS_PATH:\$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=\$PLUMED_PATH:\$LD_LIBRARY_PATH
ulimit -s unlimited

echo "ITERATION $iteration_idx"
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
    # TODO:Index file per molecule

    cp \$random_geometry $sampler_prefix.gro
    cp \$random_topol $sampler_prefix.top

    gmx grompp -f $absolute_mdp_file_path -c $sampler_prefix.gro -p $sampler_prefix.top -maxwarn 1 -o $sampler_prefix.tpr $redirect # TODO: Index file
    if [ -n "$absolute_plumed_file_path" ]
    then
        gmx mdrun -ntomp 1 -ntmpi 1 -deffnm $sampler_prefix -plumed $absolute_plumed_file_path -pin on -pinoffset \$pinoffset -pinstride 1 $redirect
    else
        gmx mdrun -ntomp 1 -ntmpi 1 -deffnm $sampler_prefix -pin on -pinoffset \$pinoffset -pinstride 1 $redirect
    fi

    if [ -f adaptive_sampling.xtc ]
    then
        echo 0 | gmx trjconv -s $sampler_prefix.tpr -f adaptive_sampling.xtc -o $ORCA_BASENAME.gro $redirect # Single entry .xtc, has to be converted directly because the .tpr might change
        echo "Sampling step \$sampler_job_idx: \$(grep step $ORCA_BASENAME.gro)" # For output in .out
        cat $ORCA_BASENAME.gro >> "adaptive_sampling.gro"
        echo "Adaptive Sampling $iteration_idx" >> "data_sources.txt"

        mkdir -p ../../orca_calculations/$ORCA_FOLDER_PREFIX\$sampler_job_idx
        mv $ORCA_BASENAME.gro ../../orca_calculations/$ORCA_FOLDER_PREFIX\$sampler_job_idx/$ORCA_BASENAME.gro
        cp $sampler_prefix.top ../../orca_calculations/$ORCA_FOLDER_PREFIX\$sampler_job_idx/topol.top
        rm adaptive_sampling.xtc
    fi

    rm $sampler_prefix.gro
    rm $sampler_prefix.top
done
EOM
    qsub $sampling_jobfile

#######################################################################
#########################END ADAPTIVE SAMPLING#########################
#######################################################################

#######################################################################
########################START ORCA CALCULATIONS########################
#######################################################################

    cd ../../orca_calculations

    batch_name_orca=orca_${iteration_idx}_$batch_idx

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
#$ -N $batch_name_orca
#$ -cwd
#$ -o $batch_out_file
#$ -e $batch_error_file
#$ -l qu=gtx
#$ -hold_jid "samp_${iteration_idx}_$batch_idx"

# Which GPU?
gpu_id=\`   echo \$QUEUE | awk '/a/ {print 0} /b/ {print 1}  /c/ {print 2}  /d/ {print 3}'\`

# How many cores are there?
case \$HOSTNAME in
    gtx0[1-6]*)
    cores=10;
    ;;
    gtx0[7-8]*)
    cores=12;
    ;;
    gtx09* | gtx10*)
    cores=16;
    ;;
    *)
    echo "Error: Unknown compute node \$HOSTNAME"
    echo "       This script only works for gtx01 thru 10!"
    echo
    exit -1
    ;;
esac

# Offset for pinning on cores
pinoffset=\`echo \$QUEUE \$cores | awk '/a/ {print 0} /b/ {print \$2} /c/ {print 2*\$2} /d/ {print 3*\$2}'\`

GROMACS_ORCA_PATH="$GROMACS_ORCA_PATH"
ORCA_PATH="$ORCA_PATH"
BLAS_PATH="$BLAS_PATH"
PLUMED_PATH="$PLUMED_PATH"

echo "JOB: $JOB_ID started on $HOSTNAME -- \$(date)"
source "$GROMACS_ORCA_PATH"
export PATH=\$ORCA_PATH:\$PATH
export LD_LIBRARY_PATH=\$BLAS_PATH:\$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=\$PLUMED_PATH:\$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=\$ORCA_PATH:\$LD_LIBRARY_PATH
export GMX_ORCA_PATH="\$ORCA_PATH"
export GMX_MAXBACKUP=-1
export GMX_QMMM_VARIANT=2
export OMP_NUM_THREADS=1
export GMX_QM_ORCA_BASENAME=$ORCA_BASENAME

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
    workdir="/scratch/$USER/orca_\${JOB_ID}_\$suffix"
    mkdir -p \$workdir

    cp topol.top \$workdir
    cp $ORCA_BASENAME.gro \$workdir
    cp ../../../$ORCA_BASENAME.mdp \$workdir
    cp ../../../$ORCA_BASENAME.ORCAINFO \$workdir
    # TODO: Index file
      
    cd \$workdir  

    echo "%PAL NPROCS \$cores END" >> $ORCA_BASENAME.ORCAINFO

    gmx grompp -f $ORCA_BASENAME.mdp -c $ORCA_BASENAME.gro -p topol.top -o $ORCA_BASENAME.tpr -maxwarn 1 $redirect
    gmx mdrun -ntomp 1 -ntmpi 1 -deffnm $ORCA_BASENAME -pin on -pinoffset \$pinoffset -pinstride 1 $redirect

    rm $ORCA_BASENAME.densities
    rm $ORCA_BASENAME.gbw

    cd \$current_dir
    rsync -a \$workdir/ .
    rm -rf \$workdir

    cd ..
done

EOM
    
    cd .. # back to root adaptive_sampling folder
done # batches of sampling and orca

# Iterate again to submit orca jobs after sampling jobs
cd orca_calculations
for batch_idx in $(seq 0 $BATCH_SIZE $((N_SAMPLES-1)))
do
    batch_idx=$(printf "%0${padding_length_samples}d" $batch_idx) # Pads with zeros
    orca_jobfile="orca_jobfile_$batch_idx.sge" 
    qsub $orca_jobfile
done
cd .. # back to root adaptive_sampling folder

#######################################################################
#########################END ORCA CALCULATIONS#########################
#######################################################################

#######################################################################
############################START RETRAINING###########################
#######################################################################
cd retraining
iteration_folder="iteration_$(printf "%0${#ITERATIONS}d" "$iteration_idx")"
mkdir -p $iteration_folder
cd $iteration_folder

retraining_jobfile="retraining_jobfile.sge"
cat << EOM > $retraining_jobfile
#!/bin/bash
#$ -N retrain_${iteration_idx}
#$ -cwd
#$ -o $out_file
#$ -e $error_file
#$ -l qu=gtx
#$ -hold_jid "orca_${iteration_idx}_*"

queue_script="/home/lpetersen/bin/qpython.sh"
preparation_script="/home/lpetersen/kgcnn_fork/prepare_data.py"
retraining_script="/home/lpetersen/kgcnn_fork/transfer_learning.py"

######## SOME MERGING OF PARALLEL DATA COLLECTION ########
cd ../../sampling
cat samp_${iteration_idx}_*/adaptive_sampling.gro >> "adaptive_sampling_$iteration_idx.gro"
cat samp_${iteration_idx}_*/data_sources.txt >> "$current_data_source_file"
cd ../retraining/$iteration_folder

cd ../../orca_calculations

# OpenMP needs this: set stack size to unlimited
ulimit -s unlimited

# Check for orca calculation failures, rename folder on failure
for job_folder in $ORCA_FOLDER_PREFIX*
do
    if !  grep -q "FINAL SINGLE" \$job_folder/$ORCA_BASENAME.out 2> /dev/null
    then
        echo "\$job_folder failed"
        mv "\$job_folder" "failed_\$job_folder" # keep folder for error message, but remove from future loops
    fi
done

# Extract orca calculations results
/home/lpetersen/bin/extract_qmmm_information.sh $AT_COUNT $ORCA_FOLDER_PREFIX $ORCA_BASENAME
cat geoms.xyz >> $current_geom_file
cat charges_hirsh.txt >> $current_charge_file
cat forces.xyz >> $current_force_file
cat esps_by_mm.txt >> $current_esp_file
cat esp_gradients.txt >> $current_esp_gradient_file

if [ -n "$ENERGY_REFERENCE_FILE" ]
then
    python3 /home/lpetersen/bin/calculate_energy_diff.py -e energies.txt -r $ENERGY_REFERENCE_FILE -o energy_diff.txt
    cat energy_diff.txt >> $current_energy_file
else
    echo "No reference energy supplied, appending raw energies!"
    cat energies.txt >> $current_energy_file
fi

################# DATA MERGING DONE ####################

\$queue_script \$preparation_script ${current_training_data_folder}/data_prep_config.json 

# if [ "\$(basename \$PWD)" == "orca_calculations" ]
# then rm -r $ORCA_FOLDER_PREFIX*
# fi

cd ../retraining/$iteration_folder

\$queue_script \$retraining_script ../retraining_config.json

EOM

qsub $retraining_jobfile

cd ../..
#######################################################################
#############################END RETRAINING############################
#######################################################################
done # iterations

cd ..