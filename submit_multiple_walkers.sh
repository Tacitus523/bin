# Give the .tpr as $1 and the plumed as $2, and any other files as $3, $4, etc.

N_WALKER=16
WALKER_SCRIPT="/home/lpetersen/bin/multiple_walkers.sh"

out_file=metadynamics.out
error_file=metadynamics.err

print_usage() {
  echo "Usage: 'submit_multiple_walkers.sh tpr_file plumed_file other_files'"
}

tpr_file=$1
shift
plumed_file=$1
shift
other_files=$@

if [ ! -f $tpr_file ]
then
    echo "tpr file not found"
    print_usage
    exit 1
fi
if [[ ! $tpr_file == *.tpr ]]; then
    echo "tpr file must end with .tpr"
    print_usage
    exit 1
fi
if [ ! -f $plumed_file ]
then
    echo "plumed file not found"
    print_usage
    exit 1
fi

if [ -f $out_file ]
then
    rm $out_file
fi
if [ -f $error_file ]
then
    rm $error_file
fi

jobname=$(basename $tpr_file .tpr) # Use the tpr file name as the job name

qsub -N $jobname -pe nproc $N_WALKER -o $out_file -e $error_file $WALKER_SCRIPT $tpr_file $plumed_file $other_files
