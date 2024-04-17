#!/bin/bash
#Give folder-prefix as $1, error-file as $2

folder_prefix="$1"
error_file="$2"

if [[ -z $1 || -z $2 ]]
then
    echo `date`" - Missing mandatory arguments:  folder-prefix or error-file"
    echo `date`" - Usage: extract_errors_iterations.sh  [folder-prefix] [error-file] . "
    exit 1
fi

target_file="errors_iterations.json"
plot_script="/lustre/home/ka/ka_ipc/ka_he8978/bin/plot_errors_iterations.py"

set -o errexit   # (or set -e) cause batch script to exit immediately when a command fails.
set -o pipefail  # cause batch script to exit immediately also when the command that failed is embedded in a pipeline
set -o nounset   # (or set -u) causes the script to treat unset variables as an error and exit immediately 
target_file_absolute_path=$(readlink -f $target_file)

echo "[" > $target_file_absolute_path

source_dir=$PWD
folders=$(find $folder_prefix* -maxdepth 1 -type d | sort -V) # Ensures numerical ordering without padded folders --> folder_0, folder_1, folder_2, ... instead of folder_0, folder_1, folder_10, ... 
n_folders=$(echo "$folders" | wc -l)

folder_index=0
follows_entry=false
for folder in $folders
do
    folder_index=$(($folder_index+1))
    cd $folder
    if [ -f $error_file ]
    then
        if $follows_entry
        then 
            echo "," >> $target_file_absolute_path
        fi
        cat $error_file >> $target_file_absolute_path
        follows_entry=true
    fi
    cd $source_dir

    if [ $folder_index -eq $n_folders ]
    then
        echo "" >> $target_file_absolute_path
        echo "]" >> $target_file_absolute_path
    fi
done

$plot_script -f $target_file

exit 0