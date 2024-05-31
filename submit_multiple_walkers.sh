# Give the .tpr as $1 and the plumed as $2, and any other files as $3, $4, etc.

WALKER_SCRIPT="/lustre/home/ka/ka_ipc/ka_he8978/bin/multiple_walkers.sh"

print_usage() {
  echo "Usage: 'submit_python_file.sh tpr_file plumed_file other_files'"
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
if [ ! -f $plumed_file ]
then
    echo "plumed file not found"
    print_usage
    exit 1
fi

sbatch $WALKER_SCRIPT $tpr_file $plumed_file $other_files
