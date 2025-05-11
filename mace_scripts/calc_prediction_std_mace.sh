#!/bin/bash
#SBATCH --job-name=std_calc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=std_calc.out
#SBATCH --error=std_calc.err
#SBATCH --gres=gpu:1

DATA_FOLDER="/lustre/work/ws/ws1/ka_he8978-dipeptide/training_data/B3LYP_aug-cc-pVTZ_vacuum"

# Self submission
if [ -z "$SLURM_JOB_ID" ]; then
    sbatch "$0" "$@"
    exit 0
fi
echo "Starting std_calc on $SLURM_JOB_NODELIST: $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

while getopts "d:" opt; do
    case $opt in
        d) DATA_FOLDER="$OPTARG" ;;
        *) echo "Usage: $0 [-d data_folder]" >&2; exit 1 ;;
    esac
done

data_folder=$(readlink -f "$DATA_FOLDER")

TRAIN_FILE="$data_folder/test.extxyz"
VALID_FILE="$data_folder/test.extxyz"
TEST_FILE="$data_folder/test.extxyz"

echo "Using test file: $TEST_FILE"

export PYTHONPATH=${PYTHONPATH}:/lustre/home/ka/ka_ipc/ka_he8978/MACE_QEq_development/mace-tools
export PYTHONPATH=${PYTHONPATH}:/lustre/home/ka/ka_ipc/ka_he8978/MACE_QEq_development/graph_longrange

python /home/ka/ka_ipc/ka_he8978/MACE_QEq_development/mace-tools/scripts/calc_prediction_std.py  \
    --name="model_test" \
    --train_file="$TRAIN_FILE" \
    --valid_file="$VALID_FILE" \
    --test_file="$TEST_FILE" \
    --energy_key="ref_energy" \
    --forces_key="ref_force" \
    --charges_key="ref_charge" \
    --esp_key="esp" \
    --esp_gradient_key="esp_gradient" \
    --formal_charges_from_data \
    --batch_size=10 \
    --valid_batch_size=10 \
    --device="cuda" \
    --prefix="model_energy_force"

