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

print_usage() {
    echo "Usage: $0 [-e number_of_epochs] [-d database] [-m model_path]" >&2
}

# Parse command line arguments for epochs and data folder
while getopts e:d:m: flag
do
    case "${flag}" in
        e) EPOCHS=${OPTARG};;
        d) DATABASE=${OPTARG};;
        m) MODEL_PATH=${OPTARG};;
        *) print_usage; exit 1;;
    esac
done

run_schnarc.py train fieldschnet \
    --cuda \
    --batch_size 8 \
    --environment_provider simple \
    --seed 42 \
    --overwrite \
    --split 1000 10 \
    --max_epochs $EPOCHS \
    --lr 1e-4 \
    --lr_patience 10 \
    --lr_decay 0.5 \
    --lr_min 1e-6 \
    --logger tensorboard \
    --log_every_n_epochs 1 \
    --tradeoffs tradeoffs.yaml \
    --phase_loss \
    --min_loss \
    --Huber \
    --features 256 \
    --interactions 6 \
    --cutoff 10.0 \
    --num_gaussians 50 \
    --n_layers 3 \
    --n_neurons 256 \
    --field qmmm \
    $DATABASE $MODEL_PATH
    #--parallel \ # For multiple GPUs
    #--split_path /path/to/split \ # For indices of prepared splits
    #--transfer /path/to/transfer \ # Previous training set used to compute mean
    #--real_socs \ # spin-orbit couplings
    #--log \ seems to log metadata from dataset, looks for spin orbit coupling indices
    #--diagonal \ # SOCs via diagonal elements
    #--inverse_nacs 0 \ # Weight NACs with inverse energies. 0 = False, default = 0
    #--L1 \ # L1 regularization
    #--order \ # Order states by energy
    #--finish \ # dynamics will only occur between S1 and S0
    #--verbose \ # verbose will print stats for each epoch