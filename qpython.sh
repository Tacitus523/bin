#!/bin/bash
#$ -l qu=gtx
#$ -cwd
#$ -o train.out
#$ -e train.err
#$ -l h_vmem=30G
##$ -q gtx01a,gtx01b,gtx01c,gtx01d,gtx02a,gtx02b,gtx02c,gtx02d,gtx03a,gtx03b,gtx03c,gtx03d,gtx05a,gtx05b,gtx05c,gtx05d,gtx06a,gtx06b,gtx06c,gtx06d,gtx09a,gtx09b,gtx09c,gtx09d,gtx10a,gtx10b,gtx10c,gtx10d

pythonfile=$1
pythonfile=${pythonfile#"/srv/nfs"}
shift  # Remove the first argument (python file) from the list of options
all_opts="$*" # Collect all options in a variable

# Which GPU?
#gpu_id=$( echo $QUEUE | awk '/a/ {print 0} /b/ {print 1}  /c/ {print 2}  /d/ {print 3}')
gpu_id=0 # we set this via CUDA_VISIBLE_DEVICES below, -g option is pretty deprecated

total_cores=$(nproc)
cores=$((total_cores / 4)) # assume 4 GPUs per node, fair share of cores

case $QUEUE in
    gtx*a)
        export CUDA_VISIBLE_DEVICES=0
        core_start=$((0*cores))
        core_end=$((1*cores-1))
        ;;
    gtx*b)
        export CUDA_VISIBLE_DEVICES=1
        core_start=$((1*cores))
        core_end=$((2*cores-1))
        ;;
    gtx*c)
        export CUDA_VISIBLE_DEVICES=2
        core_start=$((2*cores))
        core_end=$((3*cores-1))
        ;;
    gtx*d)
        export CUDA_VISIBLE_DEVICES=3
        core_start=$((3*cores))
        core_end=$((4*cores-1))
        ;;
    *)
        echo "Error: Unknown queue $QUEUE"
        echo "       This script only works for queues a, b, c, d!"
        echo
        exit -1
        ;;
esac 

# Echo important information into file
echo "# Hostname: " $HOSTNAME
echo "# SGE_QUEUE: " $SGE_QUEUE
echo "# Queue: " $QUEUE
echo "# Cores: " $cores " (total: " $total_cores ")"
echo "# Core range: " $core_start "-" $core_end
echo "# Job ID: " $JOB_ID
echo "# gpuid: " $gpu_id
echo "# Python file: " $pythonfile
echo "# Options: " $all_opts

# In case of external API usage I saved some API-keys here
if [ -f ~/.api_keys ]; then
    . ~/.api_keys
fi

# For WandB:
export WANDB_MODE=offline # no internet connection during calculation on nodes

# For data readin in kgcnn
export BABEL_DATADIR="/usr/local/run/openbabel-2.4.1"

export PATH="/home/lpetersen/anaconda_interpreter/bin:$PATH"
source /home/lpetersen/anaconda_interpreter/etc/profile.d/conda.sh
conda activate kgcnn_new

# set OpenMP parallel threads variable:
export OMP_NUM_THREADS=$cores

# OpenMP needs this: set stack size to unlimited
ulimit -s unlimited
# Start time of calculation
start=$( date "+%s" )

taskset -c $core_start-$core_end python3 $pythonfile -g $gpu_id $all_opts

# End time of calculation
end=$( date "+%s" )
# Now we calculate the time taken by the calculation
duration=$(( end - start ))
# Now we do Wibbly Wobbly Timey Wimey... Stuff
DAYS=$(( duration / 86400 ))
HOURS=$(( (duration % 86400) / 3600 ))
MINS=$(( ((duration % 86400) % 3600) / 60 ))
SECS=$(( ((duration % 86400) % 3600) % 60 ))
echo "Time taken: $DAYS days, $HOURS hours, $MINS minutes and $SECS seconds."
