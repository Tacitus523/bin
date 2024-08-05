# Give the .tpr as $1 and the plumed as $2, and any other files as $3, $4, etc.

N_WALKER=32 # Also has to be adjusted in plumed.dat
WALKER_SCRIPT="/lustre/home/ka/ka_ipc/ka_he8978/bin/multiple_walkers_justus.sh"

print_usage() {
  echo "Usage: 'submit_multiple_walkers_justus.sh tpr_file plumed_file other_files'"
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

sbatch --ntasks-per-node=$N_WALKER $WALKER_SCRIPT $tpr_file $plumed_file $other_files
