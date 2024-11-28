# Give the .tpr as $1 and the plumed as $2, and any other files as $3, $4, etc.

EMAIL="lukas.petersen@kit.edu"

N_WALKER=32 # Also has to be adjusted in plumed.dat
WALKER_SCRIPT="multiple_walkers_justus.sh"

print_usage() {
  echo "Usage: 'submit_multiple_walkers_justus.sh [-e] tpr_file plumed_file other_files'"
}

email_flag=""
while getopts "e" flag
do
    case $flag in
        e) email_flag="--mail-user=$EMAIL --mail-type=END,FAIL";;
        *) print_usage; exit 1;;
    esac
done
shift $((OPTIND - 1))

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

sbatch $email_flag --ntasks-per-node=$N_WALKER $WALKER_SCRIPT $tpr_file $plumed_file $other_files
