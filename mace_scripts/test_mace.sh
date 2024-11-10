#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=train.out
#SBATCH --error=train.err
#SBATCH --mail-user=lukas.petersen@kit.edu
#SBATCH --mail-type=END,FAIL

DATA_FOLDER="/lustre/work/ws/ws1/ka_he8978-dipeptide/training_data/B3LYP_aug-cc-pVTZ_vacuum"
#DATA_FOLDER="/lustre/work/ws/ws1/ka_he8978-thiol_disulfide/training_data/B3LYP_aug-cc-pVTZ_water"
VALID_FILE="$DATA_FOLDER/geoms.extxyz"
TEST_FILE="qm_mlmm.extxyz"
TRAIN_FILE="$TEST_FILE" # Dummy entry, not used, because this gets shuffled anyway

export PYTHONPATH=${PYTHONPATH}:/lustre/home/ka/ka_ipc/ka_he8978/MACE_QEq_development/mace-tools
export PYTHONPATH=${PYTHONPATH}:/lustre/home/ka/ka_ipc/ka_he8978/MACE_QEq_development/graph_longrange

python /home/ka/ka_ipc/ka_he8978/MACE_QEq_development/mace-tools/scripts/test_model_qeq.py  \
    --name="model_test" \
    --train_file="$TRAIN_FILE" \
    --valid_file="$VALID_FILE" \
    --test_file="$TEST_FILE" \
    --energy_key="ref_energy" \
    --forces_key="ref_force" \
    --charges_key="ref_charge" \
    --fermi_level_key="esp" \
    --external_field_key="electric_field" \
    --formal_charges_from_data \
    --batch_size=1 \
    --valid_batch_size=1 \
    --info_prefix="pred_"

