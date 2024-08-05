#!/bin/bash

TRAIN_SCRIPT="/lustre/home/ka/ka_ipc/ka_he8978/bin/mace_scripts/train_mace.sh"

job_name=$(basename $PWD)

sbatch --job-name=$job_name $TRAIN_SCRIPT