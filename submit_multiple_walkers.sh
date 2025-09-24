#!/usr/bin/bash
# Give the .tpr as $1 and the plumed as $2, and any other files as $3, $4, etc.

N_WALKER=16
WALKER_SCRIPT="/home/lpetersen/bin/multiple_walkers.sh"

out_file=metadynamics.out
error_file=metadynamics.err

function print_usage() {
    echo "Usage: $0 -t TPR_FILE.tpr [-p PLUMED_FILE.dat] [additional_files...]"
    exit 1
}

function remove_if_exists() {
    if [ -f "$1" ]; then
        rm "$1"
    fi
}

while getopts ":t:p:" opt; do
    case ${opt} in
        t )
            tpr_file=$OPTARG
            ;;
        p )
            plumed_file=$OPTARG
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
shift $((OPTIND -1))

# Check if mandatory argument is set
if [ -z "${tpr_file}" ]; then
    echo "Missing .tpr file" 1>&2
    print_usage
    exit 1
fi

# Validate that the tpr file exists
if [ ! -f "${tpr_file}" ]; then
    echo "TPR file ${tpr_file} does not exist" 1>&2
    exit 1
fi
echo "tpr_file: $tpr_file"

if [ -z "$plumed_file" ]
then
    echo "Missing plumed file"
    plumed_flag=""
else
    echo "plumed_file: $plumed_file"
    plumed_flag="-p $plumed_file"
fi

# Remaining arguments are additional files
additional_files=("$@")
for file in "${additional_files[@]}"; do
    echo "Additional file: $file"
done

remove_if_exists $out_file
remove_if_exists $error_file

jobname=$(basename $tpr_file .tpr) # Use the tpr file name as the job name

if [ "$HOSTNAME" == "hydrogen2" ]
then
    qsub -N $jobname -pe nproc $N_WALKER -o $out_file -e $error_file $WALKER_SCRIPT -t $tpr_file $plumed_flag "${additional_files[@]}"
elif [ "$HOSTNAME" == "tcbserver2" ]
then
    qsub -N $jobname -v N_WALKER=$N_WALKER -o $out_file -e $error_file -l qu=gtx $WALKER_SCRIPT -t $tpr_file $plumed_flag "${additional_files[@]}"
else
    echo "Unknown host: $HOSTNAME. Cannot submit job."
    exit 1
fi