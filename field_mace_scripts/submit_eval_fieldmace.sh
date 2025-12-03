#!/bin/bash
#SBATCH --job-name=eval_fieldmace
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=eval.out
#SBATCH --error=eval.err
#SBATCH --gres=gpu:1

DATA_FOLDER=""
MODEL_FILE="FieldMace_stagetwo.model"
TEST_FILE="test.xyz"
OUTPUT_FILE="model_geoms.extxyz"
CONFIG_FILE="config.yaml"
MAIL="$MY_MAIL"
model_file="FieldMace_stagetwo.model"

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
        m) model_file=${OPTARG};;
        c) CONFIG_FILE=${OPTARG};;
        *) print_usage; exit 1;;
    esac
done

# Initialize flag variables
data_folder_flag=""
config_file_flag=""

# If DATA_FOLDER not provided, try to read test_file from config
if [ -z "$DATA_FOLDER" ]; then
    echo "Data folder not specified. Please provide a data folder."
    if [ -z "$CONFIG_FILE" ]; then
        echo "Error: Either -d (data_folder) or -c (config_file) must be provided" >&2
        print_usage
        exit 1
    fi

    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Error: Config file not found: $CONFIG_FILE" >&2
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

if [ -z "$model_file" ]
then
    model_name=$(yq e '.name' $CONFIG_FILE)
    model_file="${model_name}_stagetwo.model"
    if ! [ -f "$model_file" ]
    then
        echo "SWA model file not found: $model_file" >&2
        echo "Falling back to regular model file." >&2
        model_file="${model_name}.model"
    fi
fi

if [ ! -f "$model_file" ]
then
    echo "Model file not found: $model_file" >&2
    exit 1
fi
model_file_flag="-m $model_file"

EVAL_SCRIPT=$(which mace_eval_configs)
if [ -z "$EVAL_SCRIPT" ]; then
    echo "mace_eval_configs not found in PATH. Please check your environment."
    exit 1
fi

PLOT_SCRIPT=$(which FieldMacePlot.py)
if [ -z "$PLOT_SCRIPT" ]; then
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

echo "Starting evaluation on $SLURM_JOB_NODELIST: $(date)"

$EVAL_SCRIPT \
    --configs="$test_file" \
    --batch_size=8 \
    --model=$model_file \
    --output="$OUTPUT_FILE" \
    --device="cuda" \
    --info_prefix="pred_"

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