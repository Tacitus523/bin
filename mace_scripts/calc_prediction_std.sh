#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=std_calc.out
#SBATCH --error=std_calc.out
#SBATCH --gres=gpu:1

TEST_FILE="/lustre/work/ws/ws1/ka_he8978-dipeptide/training_data/B3LYP_aug-cc-pVTZ_vacuum/test.extxyz"

calculation_script="/lustre/home/ka/ka_ipc/ka_he8978/MACE_QEq_development/mace-tools/scripts/calc_prediction_std.py"

export PYTHONPATH=${PYTHONPATH}:"/lustre/home/ka/ka_ipc/ka_he8978/MACE_QEq_development/mace-tools"
export PYTHONPATH=${PYTHONPATH}:"/lustre/home/ka/ka_ipc/ka_he8978/MACE_QEq_development/graph_longrange"

$calculation_script \
    --prefix "model_energy_force"\
    --test_file $TEST_FILE \
    --device "cuda" 