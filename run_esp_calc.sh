#!/bin/bash
#$ -N esp_calc
#$ -cwd
#$ -o esp_calc.o$JOB_ID
#$ -e esp_calc.e$JOB_ID

# folder-prefix as $1, file-prefix as $2

#CONDA_DIR="$HOME/miniconda3"
ESPS_BY_MM_FILE="esps_by_mm.txt" # file name hardcoded like this in esp_calculation_script
ESP_GRADIENT_FILE="esp_gradients.txt" # file name hardcoded like this in esp_calculation_script

esp_calculation_script="esp_calculation_from_pc.py"

if [ -f $ESPS_BY_MM_FILE ]
then rm $ESPS_BY_MM_FILE
fi

if [ -f $ESP_GRADIENT_FILE ]
then rm $ESP_GRADIENT_FILE
fi

# CONDA_DIR=$(dirname $(dirname $CONDA_EXE))
# export PATH="$CONDA_DIR/bin:$PATH"
# source $CONDA_DIR/etc/profile.d/conda.sh
# conda activate kgcnn
# PYTHONPATH=$PWD:$PYTHONPATH

echo "Start ESP calculation `date`"
python3 $esp_calculation_script --dir $1 --input $2 --unit V # ESP in Volt, change to au, if requiered
# python3 $esp_calculation_script --dir $1 --input $2 --unit V # ESP in Volt, change to au, if requiered
echo "End ESP calculation `date`"

source_dir=$PWD
