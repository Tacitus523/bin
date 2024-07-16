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

export PYTHONPATH=${PYTHONPATH}:/lustre/home/ka/ka_ipc/ka_he8978/MACE_QEq_development/mace-tools
export PYTHONPATH=${PYTHONPATH}:/lustre/home/ka/ka_ipc/ka_he8978/MACE_QEq_development/graph_longrange

/lustre/home/ka/ka_ipc/ka_he8978/miniconda3/envs/mace_env/bin/python3.12 /lustre/home/ka/ka_ipc/ka_he8978/MACE_QEq_development/mace-tools/scripts/martin_train.py  \
    --name="QEqWwater" \
    --train_file="/lustre/work/ws/ws1/ka_he8978-dipeptide/training_data/B3LYP_aug-cc-pVTZ_vacuum/geoms.extxyz" \
    --batch_size=10 \
    --valid_batch_size=20 \
    --eval_interval=1 \
    --valid_fraction=0.05 \
    --config_type_weights='{"Default":1.0}' \
    --E0s='{1: -13.575035506869515, 6: -1029.6173622986487, 7: -1485.1410643783852, 8: -2042.617308911902, 16: -10832.265333248919}' \
    --model="maceQEq" \
    --hidden_irreps='64x0e+64x1o' \
    --r_max=8.0 \
    --max_num_epochs=50 \
    --device=cuda \
    --loss="charges_energy_forces" \
    --formal_charges_from_data \
    --charges_key="charge" \
    --error_table="EFQRMSE" \
    --scale_atsize=1.0 \
    --charges_weight=100 \
    --restart_latest \
    --results_dir="results"\
    --save_cpu \
    --wandb \
    --wandb_project='Dipeptid' \
    --wandb_entity='ml4elstner' \
    --wandb_name='mace_02' \
    --wandb_log_hypers lr lr_factor lr_scheduler_gamma batch_size max_num_epochs energy_weight forces_weight charges_weight r_max hidden_irreps MLP_irreps valid_fraction
