#!/bin/bash
#$ -cwd

FOLDER_PREFIX="GEOM_"
ORCA_BASENAME="sp"
JOBFILE_NAME="job_file.sge"

GROMACS_DFTB_PATH="/home/tkubar/GMX-DFTB/gromacs-dftbplus/release-machine-learning/bin/GMXRC"
GROMACS_ORCA_PATH="/usr/local/run/gromacs-orca-avx2/bin/GMXRC"
ORCA_PATH="/usr/local/run/orca_5_0_3_linux_x86-64_shared_openmpi411"
ORCA_2MKL="/usr/local/run/orca_5_0_3_linux_x86-64_shared_openmpi411/orca_2mkl"
BLAS_PATH="/usr/local/run/OpenBLAS-0.3.20-be/lib"
PLUMED_PATH="/usr/local/run/plumed-2.5.1-openblas/lib"
MULTIWFN_PATH="/home/lpetersen/Multiwfn/Multiwfn_3.8_dev_bin_Linux_noGUI/Multiwfn"

lower_limit=$1
upper_limit=$2
padding_size=$3

echo "JOB: $JOB_ID started on $HOSTNAME -- $(date)"
source $GROMACS_ORCA_PATH
export PATH=$ORCA_PATH:$PATH
export LD_LIBRARY_PATH=$BLAS_PATH:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$PLUMED_PATH:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$ORCA_PATH:$LD_LIBRARY_PATH
export GMX_ORCA_PATH=$ORCA_PATH
export GMX_QMMM_VARIANT=2
export OMP_NUM_THREADS=1
export GMX_DFTB_ESP=1
export GMX_QM_ORCA_BASENAME=$ORCA_BASENAME

ulimit -s unlimited

for suffix in $(seq $lower_limit $upper_limit)
do
    echo "JOB: Geometry $lower_limit..$suffix..$upper_limit -- $(date)"
    padded_suffix=$(printf "%0${padding_size}d" $suffix)
    folder="${FOLDER_PREFIX}${padded_suffix}"
    cd $folder
    bash $JOBFILE_NAME
    cd ..
done
