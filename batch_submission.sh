#!/usr/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 start end batch_size [options]"
    echo ""
    echo "Arguments:"
    echo "  start           Start index"
    echo "  end             End index"
    echo "  batch_size      Size of each batch"
    echo ""
    echo "Options:"
    echo "  -p, --padding   Padding size (default: length of end parameter)"
    echo "  -b, --batch-file Batch file to execute (default: batch.sh)"
    echo "  -h, --help      Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 1 100 10 --padding 3 --batch-file custom_batch.sh"
    exit 1
}

# Parse arguments
POSITIONAL_ARGS=()
batch_file="batch.sh"
padding_size=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--padding)
            padding_size="$2"
            shift 2
            ;;
        -b|--batch-file)
            batch_file="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        -*)
            echo "Unknown option $1"
            usage
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Restore positional parameters
set -- "${POSITIONAL_ARGS[@]}"

# Check for required arguments
if [[ -z $1 || -z $2 || -z $3 ]]; then
    echo "$(date) - Missing mandatory arguments: start, end, batch_size"
    usage
elif [[ $# -gt 3 ]]; then
    echo "$(date) - Too many arguments provided"
    usage
fi

set -o errexit # Exit immediately when a command fails

start=$1
end=$2
batch_size=$3

# Set default padding size if not provided
if [[ -z "$padding_size" ]]; then
    padding_size=$((${#end}))
fi

# Validate padding size
if [[ $padding_size -lt ${#end} ]]; then
    echo "Requested padding is $padding_size, which is less than the required padding for the end $end of ${#end}."
    echo "Therefore no padding will be performed."
    padding_size=0
fi

# Check if batch file exists
if [[ ! -f "$batch_file" ]]; then
    echo "$(date) - Error: Batch file '$batch_file' not found"
    exit 1
fi

echo "$(date) - Starting batch submission with batch file: $batch_file"

for i in $(seq $start $batch_size $end)
do
    lower_limit=$i
    upper_limit=$((i+batch_size-1))
    if [ $upper_limit -gt $end ]
    then
        upper_limit=$end
    fi

    batch_name=batch_$(printf "%0${padding_size}d" $lower_limit)

    if [ -f $batch_name.err ]
    then rm $batch_name.err
    fi

    if [ -f $batch_name.out ]
    then rm $batch_name.out
    fi

    qsub -N $batch_name -o $batch_name.out -e $batch_name.err $batch_file $lower_limit $upper_limit $padding_size
done

