# Give the .tpr as $1 and the plumed as $2, and any other files as $3, $4, etc.

EMAIL="lukas.petersen@kit.edu"

N_WALKER=16 # Also has to be adjusted in plumed.dat
WALKER_SCRIPT="multiple_walkers_justus.sh"

print_usage() {
    echo "Usage: $0 -t TPR_FILE.tpr [-p PLUMED_FILE.dat] [-e] [additional_files...]"
    exit 1
}

while getopts ":t:p:e" opt; do
    case ${opt} in
        t )
            tpr_file=$OPTARG
            ;;
        p )
            plumed_file=$OPTARG
            ;;
        e )
            email_flag="--mail-user=$EMAIL --mail-type=END,FAIL"
            ;;
        \? )
            echo "Invalid option: -$OPTARG" 1>&2
            print_usage
            ;;
        : )
            echo "Invalid option: -$OPTARG requires an argument" 1>&2
            print_usage
            ;;
    esac
done
shift $((OPTIND - 1))

# Check if mandatory argument is set
if [[ ! $tpr_file == *.tpr ]]; then
    echo "tpr file must end with .tpr. Got $tpr_file"
    print_usage
    exit 1
fi
echo "tpr_file: $tpr_file"
if [ ! -f $plumed_file ]
then
    echo "plumed file not found. Got $plumed_file"
    print_usage
    exit 1
fi
echo "plumed_file: $plumed_file"

# Remaining arguments are additional files
additional_files=("$@")
for file in "${additional_files[@]}"; do
    echo "Additional file: $file"
done

job_name=$(basename $tpr_file .tpr)_metaD

sbatch $email_flag --ntasks-per-node=$N_WALKER --job-name=$job_name $WALKER_SCRIPT -t $tpr_file -p $plumed_file "${additional_files[@]}"
