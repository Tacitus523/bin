#!/usr/bin/bash

set -e
set -u

if [ $# -eq 0 ]; then
  echo "No additional arguments provided. At least the config file is required."
  exit 1
fi

experiment_name=$(basename $1 .yaml)
config_dir=$(dirname $(dirname $(realpath $1)))
shift

if [ ! -d "$config_dir" ]; then
  echo "Error: Config directory '$config_dir' does not exist."
  exit 1
fi

if [ ! -f "$0" ]; then
  echo "Error: Config file '$1' does not exist."
  exit 1
fi

spk_train_file=$(which spktrain)

submit_python_file_justus.sh -e "$spk_train_file" -- --config-dir="$config_dir" experiment="$experiment_name" "$@"