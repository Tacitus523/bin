# Give the .tpr as $1 and the plumed as $2, and any other files as $3, $4, etc.

EMAIL="lukas.petersen@kit.edu"

WALKER_SCRIPT="single_walker_justus.sh"

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

if [ -z "$plumed_file" ]
then
    echo "Missing plumed file"
    plumed_command=""
else
    echo "plumed_file: $plumed_file"
    plumed_command="-p $plumed_file"
fi

# Remaining arguments are additional files
additional_files=("$@")
for file in "${additional_files[@]}"; do
    echo "Additional file: $file"
done

job_name=$(basename $tpr_file .tpr)_sim

sbatch $email_flag --ntasks-per-node=1 --job-name=$job_name $WALKER_SCRIPT -t $tpr_file $plumed_command $additional_files
