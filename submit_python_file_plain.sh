#!/usr/bin/env python3
import os
import argparse

ap = argparse.ArgumentParser(description="Submit a python-script to the queue")
ap.add_argument("-i", "--input", type=str, dest="script", action="store", required=True, help="Python script to submit", metavar="script-path")
ap.add_argument("-n", "--name", default="python-script", type=str, dest="name", action="store", required=False, help="Name in queue", metavar="name")
ap.add_argument("-e", "--env", default="venv_tf_new", type=str, dest="env", action="store", required=False, help="Conda environment to activate", metavar="environment")
args = ap.parse_args()

script = args.script
job_name = args.name
environment = args.env

job_file = "job_file.sge"
cwd = os.getcwd()
# cuts /srv/nfs/ of path
if cwd.split("/")[1]=="srv":
    cwd="/"+cwd.split("/",3)[3]

cores = 1 #doesn't seem to profit from more cores

job_string = f'''#!/bin/bash
#$ -N {job_name}
#$ -cwd
#$ -o $JOB_NAME.o$JOB_ID
#$ -e $JOB_NAME.e$JOB_ID
#$ -pe nproc {cores}
export OMP_NUM_THREADS={cores}
echo "JOB $JOB_NAME: $JOB_ID started on $HOSTNAME -- $(date)"

ulimit -s unlimited

export PATH="~/miniconda3/bin:$PATH"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate {environment}
PYTHONPATH=$PWD:$PYTHONPATH
python3 $PWD/{script}

/bin/rm -rf {job_file}
'''

with open(job_file, "w") as f:
	f.write(job_string)

os.system(f"qsub {job_file}")
