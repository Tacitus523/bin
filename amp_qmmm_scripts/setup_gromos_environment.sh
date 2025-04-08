#!/bin/bash
#  Sets up the environment for Gromos-commands, use 'source setup_gromos_environment.sh' to run this script

module load devel/oneapi/2024.2.1
module load mpi/latest
module load numlib/gsl/2.6
module load lib/cudnn/9.0.0_cuda-12.3

#conda activate amp_qmmm
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/$HOME/sw/fftw3_mpi/lib/"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/sw/torch-extensions/lib64"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/sw/xtb-6.5.1/lib64/"