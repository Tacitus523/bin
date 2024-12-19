#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=train.out
#SBATCH --error=train.err
#SBATCH --gres=gpu:1
#SBATCH --mail-user=lukas.petersen@kit.edu
#SBATCH --mail-type=END,FAIL

export PYTHONPATH=${PYTHONPATH}:"/lustre/home/ka/ka_ipc/ka_he8978/MACE_QEq_development/mace-tools"
export PYTHONPATH=${PYTHONPATH}:"/lustre/home/ka/ka_ipc/ka_he8978/MACE_QEq_development/graph_longrange"

# Relative paths in the data folder
TRAIN_FILE="train.extxyz"
VALID_FILE="valid.extxyz"
TEST_FILE="test.extxyz"
# Default value, gets overwritten by the command line argument from submit_train_mace.sh
EPOCHS=100

MODEL_NAME="QEq"
MODEL_TYPE="maceQEq"
#MODEL_TYPE="Polarizable"


# Just if you want to use wandb
WANDB_PROJECT="Dipeptid"
# WANDB_PROJECT="Thioldisulfide_Water"
WANDB_ENTITY="ml4elstner" # Gets submitted into the group folder like this
if [ -z $SLURM_JOB_NAME ]; then
    WANDB_NAME=$(basename $PWD)
else
    WANDB_NAME=$SLURM_JOB_NAME
fi

print_usage() {
    echo "Usage: $0 [-e number_of_epochs] [-d data_folder]" >&2
}

# Parse command line arguments for epochs and data folder
while getopts e:d: flag
do
    case "${flag}" in
        e) EPOCHS=${OPTARG};;
        d) DATA_FOLDER=${OPTARG};;
        *) print_usage; exit 1;;
    esac
done

data_folder=$(realpath $DATA_FOLDER)
train_file="$data_folder/$TRAIN_FILE"
valid_file="$data_folder/$VALID_FILE"
test_file="$data_folder/$TEST_FILE"

echo "Starting training: $(date)"
echo "Data folder: $data_folder"
echo "Train file: $train_file"
echo "Valid file: $valid_file"
echo "Test file: $test_file"
echo "Model name: $MODEL_NAME"
echo "Model type: $MODEL_TYPE"
echo "Number of epochs: $EPOCHS"

python /lustre/home/ka/ka_ipc/ka_he8978/MACE_QEq_development/mace-tools/scripts/lukas_train.py  \
    --name=$MODEL_NAME \
    --train_file=$train_file \
    --valid_file=$valid_file \
    --test_file=$test_file \
    --batch_size=10 \
    --valid_batch_size=10 \
    --eval_interval=1 \
    --config_type_weights='{"Default":1.0}' \
    --E0s='{1: -13.575035506869515, 6: -1029.6173622986487, 7: -1485.1410643783852, 8: -2042.617308911902, 16: -10832.265333248919}' \
    --model=$MODEL_TYPE \
    --hidden_irreps='64x0e+64x1o' \
    --r_max=8.0 \
    --max_num_epochs=$EPOCHS \
    --device=cuda \
    --loss="charges_energy_forces" \
    --energy_key="ref_energy" \
    --forces_key="ref_force" \
    --charges_key="ref_charge" \
    --esp_key="esp" \
    --esp_gradient_key="esp_gradients" \
    --formal_charges_from_data \
    --error_table="EFQRMSE" \
    --scale_atsize=1.0 \
    --energy_weight=1 \
    --forces_weight=100 \
    --charges_weight=50 \
    --swa \
    --swa_energy_weight=1000 \
    --swa_forces_weight=100 \
    --ema \
    --ema_decay=0.99 \
    --restart_latest \
    --results_dir="results" \
    --save_cpu \
    --wandb \
    --wandb_project=$WANDB_PROJECT \
    --wandb_entity=$WANDB_ENTITY \
    --wandb_name=$WANDB_NAME \
    --wandb_log_hypers lr lr_factor lr_scheduler_gamma batch_size max_num_epochs energy_weight forces_weight charges_weight r_max hidden_irreps MLP_irreps valid_fraction
    # --kspace_cutoff_factor 1.5 \ # Polarizable model only
    # --atomic_multipoles_max_l 0 \ # Polarizable model only
    # --atomic_multipoles_smearing_width 1.0 \ # Polarizable model only
    # --field_feature_widths 1.0 \ # Polarizable model only
    # --include_electrostatic_self_interaction \ # Polarizable model only
    # --include_local_electron_energy \ # Polarizable model only
    # --field_dependence_type "local_linear" \ # Polarizable model only
    # --final_field_readout_type "StrictQuadraticFieldEnergyReadout" \ # Polarizable model only
    # --quadrupole_feature_corrections \ # Polarizable model only
    # --valid_fraction=0.05 \ # Retired with the introduction of the valid.extxyz file
    # --start_swa=450 \ # Default is last 20% of epochs, which seems simpler to use

training_exit_status=$?

echo "Finished training: $(date)"

# Convert the model to a scripted model
if [ $training_exit_status -eq 0 ]
then
    convert_model_to_scripted_model.py --model_prefix $MODEL_NAME
fi
