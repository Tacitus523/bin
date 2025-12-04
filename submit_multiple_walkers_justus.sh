# Give the .tpr as $1 and the plumed as $2, and any other files as $3, $4, etc.

EMAIL=$MY_MAIL

export N_WALKER=16 # Will be read from plumed file if provided, default to 16 otherwise
WALKER_SCRIPT="multiple_walkers_justus.sh"

print_usage() {
    echo "Usage: $0 -t TPR_FILE.tpr [-p PLUMED_FILE.dat] [-e] [additional_files...]"
    exit 1
}

resource_flag="--gres=gpu:1"

email_flag="--mail-user=$EMAIL --mail-type=END,FAIL"
while getopts ":t:p:e:g:" opt; do
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
        g )
            if [[ ! $OPTARG =~ ^[0-9]+$ ]]; then
                echo "Invalid GPU count: $OPTARG"
                print_usage
                exit 1
            fi
            if [ $OPTARG -eq 0 ]; then
                resource_flag=""
            else
                resource_flag="--gres=gpu:$OPTARG"
            fi
            echo "Requesting $OPTARG GPU(s) per node"
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
elif [ ! -f "$tpr_file" ]; then
    echo "tpr file not found. Got $tpr_file"
    print_usage
    exit 1
fi
echo "tpr_file: $tpr_file"

if [ -z "$plumed_file" ]; then
    echo "WARNING: Missing plumed file"
elif [ ! -f "$plumed_file" ]; then
    echo "plumed file not found. Got $plumed_file"
    print_usage
    exit 1
else
    export N_WALKER=$(grep "WALKERS_N=" $plumed_file | cut -d'=' -f2)
    echo "plumed_file: $plumed_file"
fi

# Remaining arguments are additional files
additional_files=("$@")
for file in "${additional_files[@]}"; do
    echo "Additional file: $file"
done

job_name=$(basename $tpr_file .tpr)_metaD

N_TASKS=$N_WALKER
sbatch $email_flag --ntasks-per-node=$N_TASKS --job-name=$job_name $resource_flag $WALKER_SCRIPT -t $tpr_file -p $plumed_file "${additional_files[@]}"
