#!/bin/bash
#SBATCH --job-name=eval_mace_qEq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=eval.out
#SBATCH --error=eval.err
##SBATCH --gres=gpu:1

DATA_FOLDER=""
TEST_FILE="test.extxyz"
MODEL_FILE="QEq_swa.model"
OUTPUT_FILE="model_geoms.extxyz"
CONFIG_FILE="config.yaml"
MAIL="$MY_MAIL"

print_usage() {
    echo "Usage: $0 [-d data_folder] [-m model_file] [-c config_file]" >&2
    echo "  -d: Data folder containing test file" >&2
    echo "  -m: Model file to use for evaluation" >&2
    echo "  -c: YAML config file to read test_file path from (used if -d not given)" >&2
}

while getopts ":d:m:c:" flag
do
    case "${flag}" in
        d) DATA_FOLDER=${OPTARG};;
        m) MODEL_FILE=${OPTARG};;
        c) CONFIG_FILE=${OPTARG};;
        *) print_usage; exit 1;;
    esac
done

# Initialize flag variables
data_folder_flag=""
config_file_flag=""

# If DATA_FOLDER not provided, try to read test_file from config
if [ -z "$DATA_FOLDER" ]; then
    if [ -z "$CONFIG_FILE" ]; then
        echo "Error: Either -d (data_folder) or -c (config_file) must be provided" >&2
        print_usage
        exit 1
    fi
    
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Error: Config file not found: $CONFIG_FILE" >&2
        exit 1
    fi
    
    # Check if yq is available
    if ! command -v yq &> /dev/null; then
        echo "Error: yq command not found. Please install yq to use -c option without -d" >&2
        echo "  Install with: pip install yq" >&2
        exit 1
    fi
    
    # Read test_file from YAML config
    test_file=$(yq eval '.test_file' "$CONFIG_FILE")
    
    if [ -z "$test_file" ] || [ "$test_file" = "null" ]; then
        echo "Error: test_file not found in config file: $CONFIG_FILE" >&2
        exit 1
    fi
    
    echo "Read test_file from config."
    config_file_flag="-c $CONFIG_FILE"
else
    data_folder_flag="-d $DATA_FOLDER"
    test_file=$(readlink -f $DATA_FOLDER/$TEST_FILE)
fi

if [ ! -f "$test_file" ]
then
    echo "Error: Test file not found: $test_file" >&2
    exit 1
fi

model_file=$MODEL_FILE
model_file_flag="-m $model_file"
if [ ! -f "$model_file" ]
then
    echo "Model file not found: $model_file" >&2
    exit 1
fi

PLOT_SCRIPT=$(which MacePlot.py)
if [ -z "$PLOT_SCRIPT" ]
then
    echo "MacePlot.py not found in PATH. Please check your environment." >&2
    exit 1
fi

echo "Using test file: $test_file"
echo "Using model file: $model_file"

# Check if the script is being run in a SLURM job
# If not, submit it as a job
if [ -z "$SLURM_JOB_ID" ]
then
    mail_flag="--mail-user=$MAIL --mail-type=END,FAIL"
    eval_output=$(sbatch --parsable $mail_flag $0 $data_folder_flag $model_file_flag $config_file_flag)
    echo "Submitted evaluation job with ID: $eval_output"
    exit
fi

export PYTHONPATH=${PYTHONPATH}:"/lustre/home/ka/ka_ipc/ka_he8978/MACE_QEq_development/mace-tools"
export PYTHONPATH=${PYTHONPATH}:"/lustre/home/ka/ka_ipc/ka_he8978/MACE_QEq_development/graph_longrange"

echo "Starting evaluation on $SLURM_JOB_NODELIST: $(date)"

python /lustre/home/ka/ka_ipc/ka_he8978/MACE_QEq_development/mace-tools/scripts/eval_qeq.py \
        --configs="$test_file" \
        --model="$model_file" \
        --output="$OUTPUT_FILE" \
        --device="cpu"
eval_exit_status=$?

if [ $eval_exit_status -ne 0 ]
then
    echo "Evaluation failed with exit status $eval_exit_status" >&2
    exit $eval_exit_status
else
    echo "Evaluation completed successfully: $(date)"
fi
echo "Output written to: $OUTPUT_FILE"

echo "Starting plot generation: $(date)"
$PLOT_SCRIPT -g $OUTPUT_FILE
plot_exit_status=$?
if [ $plot_exit_status -ne 0 ]
then
    echo "Plot generation failed with exit status $plot_exit_status" >&2
    exit $plot_exit_status
else
    echo "Plot generation completed successfully: $(date)"
fi