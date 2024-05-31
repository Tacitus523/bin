#!/bin/bash
#Give folder-prefix as $1, out-file as $2

folder_prefix="$1" # samp_
out_file="$2" # sampling.out

if [[ -z $1 || -z $2 ]]
then
    echo `date`" - Missing mandatory arguments:  folder-prefix or out-file"
    echo `date`" - Usage: extract_n_timesteps.sh  [folder-prefix] [out-file] . "
    exit 1
fi

target_file="timesteps.csv"
search_term="step"
plot_script="/lustre/home/ka/ka_ipc/ka_he8978/bin/plot_boxplot_timesteps.py"

set -o errexit   # (or set -e) cause batch script to exit immediately when a command fails.
source_dir=$PWD
target_file_absolute_path=$(readlink -f $target_file)
echo "Iteration;Timestep" > $target_file_absolute_path

folders=$(find $folder_prefix* -maxdepth 1 -type d | sort -V) # Ensures numerical ordering without padded folders --> folder_0, folder_1, folder_2, ... instead of folder_0, folder_1, folder_10, ... 
for folder in $folders
do
    cd $folder
    folder_basename=$(basename $folder)
    iteration_index="${folder_basename#*_}"     # Remove everything before the first underscore
    iteration_index="${iteration_index%_*}"     # Remove everything after the last underscore
    if [ -f $out_file ]
    then
        time_steps=$(grep $search_term $out_file | awk '{print $NF}')
        for time_step in $time_steps
        do
            echo "Iteration $iteration_index;$time_step" >> $target_file_absolute_path
        done
    fi
    cd $source_dir
done

$plot_script -f $target_file