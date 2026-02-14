#!/usr/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --time=200:00:00
#SBATCH --output=train.out
#SBATCH --error=train.err
#SBATCH --gres=gpu:1

if [ $# -eq 0 ]; then
  echo "No additional arguments provided. At least the config file is required."
  exit 1
fi

config_yaml=$1
config_name=$(basename $config_yaml .yaml)
config_dir=$(dirname $(realpath $config_yaml))
shift

if [ ! -d "$config_dir" ]; then
  echo "Error: Config directory '$config_dir' does not exist."
  exit 1
fi

if [ ! -f "$0" ]; then
  echo "Error: Config file '$1' does not exist."
  exit 1
fi

# Check if the script is being run in a SLURM job
# If not, submit it as a job
if [ -z "$SLURM_JOB_ID" ]
then
    name_flag="--job-name=spkpredict_${config_name}"
    email_flag="--mail-user=$MY_MAIL --mail-type=END,FAIL"
    
    job_id=$(sbatch --parsable $name_flag $email_flag "$0" "$config_yaml" "$@")
    echo "Submitted prediction job with ID: $job_id"
    exit 0
fi

datapath=$(yq e '.run.data_dir' "$config_yaml")/$(yq e '.globals.db_name' "$config_yaml")
model_dir=$config_dir
model_path=$(yq e '.globals.model_path' "$config_yaml")
cutoff=$(yq e '.globals.cutoff' "$config_yaml")
output_dir=prediction

# Create symlink to the best model if it doesn't exist, name used to be hardcoded in spkpredict
if [ ! -e "best_model" ]; then
  ln -s "$model_path" best_model
fi

if [ -d "$output_dir" ]; then
  rm -rf "$output_dir"
fi

python_execution_script=qpython_justus.sh
python_script="$(which spkpredict)"
$python_execution_script $python_script datapath=$datapath modeldir=$model_dir modelname=$model_path outputdir=$output_dir cutoff=$cutoff enable_grad=true split_file="split.npz" write_idx_m=true "$@"

if [ $? -ne 0 ]; then
    echo "Error: Prediction script failed."
    exit 1
fi

cd $output_dir
SchnetPackPlot.py 
