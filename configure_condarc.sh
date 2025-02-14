#!/usr/bin/bash

set -e # Exit on error
set -u # Exit on using unset variable

which conda
if [ $? -ne 0 ]; then
    echo "Conda not found. Please install conda and try again."
    exit 1
fi

NEW_CONDA_ENVS=/home/conda/envs
NEW_CONDA_PKGS=/home/conda/pkgs

CONDA_HOME=$(dirname $(dirname $(which conda)))
CONDA_ENVS=$CONDA_HOME/envs
CONDA_PKGS=$CONDA_HOME/pkgs

echo "Found conda home: $CONDA_HOME"
echo "Assuming old conda envs: $CONDA_ENVS"
echo "Assuming old conda pkgs: $CONDA_PKGS"

echo "Adding new conda envs: $NEW_CONDA_ENVS"
conda config --add envs_dirs $CONDA_ENVS
conda config --add envs_dirs $NEW_CONDA_ENVS # Takes precedence over old envs

echo "Adding new conda pkgs: $NEW_CONDA_PKGS"
conda config --add pkgs_dirs $CONDA_PKGS
conda config --add pkgs_dirs $NEW_CONDA_PKGS # Takes precedence over old pkgs

