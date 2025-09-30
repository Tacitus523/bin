# Submit the python script to the queue and gives it the name of the current folder in the queue
# Use the -s flag to keep this process running until it finishes and sync training data to wandb
# Use the -c flag to submit a python file with a configuration file
# Use the -p flag to specify the python file to run
# Use the -e flag to send an email when the job finishes or fails

EMAIL=$MY_MAIL

queue_script="$HOME/bin/qpython_justus.sh"

print_usage() {
  echo "Usage: submit_python_file_justus.sh [-e] python_file.py [-- additional_args]"
  echo "  -e             : Send email when job finishes or fails (optional)"
  echo "  -j n_tasks    : Number of tasks per node (optional, default is 1)"
  echo "  python_file.py : Python file to submit (mandatory)"
  echo "  additional_args: Any additional arguments to pass to the submitted job"
}

# Parse options and arguments
email_flag=""
ntasks_flag=""
ngpu_flag="--gres=gpu:1"  # Default to 1 GPU
python_script=""
additional_args=""

# Process all arguments to separate flags from script name and additional args
while [ $# -gt 0 ]; do
  case $1 in
    -e)
      email_flag="--mail-user=$EMAIL --mail-type=END,FAIL"
      echo "Email notifications enabled for $EMAIL"
      shift
      ;;
    -j)
      n_tasks=$2
      ntasks_flag="--ntasks-per-node=$n_tasks"
      echo "Number of tasks per node set to $n_tasks"
      shift
      shift
      ;;
    -g)
      n_gpus=$2
      if [ "$n_gpus" -gt 0 ]; then
        ngpu_flag="--gres=gpu:$n_gpus"
        echo "Number of GPUs set to $n_gpus"
      elif [ "$n_gpus" -eq 0 ]; then
        ngpu_flag=""
        echo "No GPUs will be requested"
      else
        echo "Error: Number of GPUs must be a non-negative integer."
        exit 1
      fi
      shift
      shift
      ;;
    -h)
      print_usage
      exit 0
      ;;
    --)
      shift
      additional_args="$*"
      echo "Additional arguments: $additional_args"
      break
      ;;
    -*)
      echo "Error: Unknown option $1"
      print_usage
      exit 1
      ;;
    *)
      if [ -z "$python_script" ]; then
        python_script="$1"
        shift
      else
        # All remaining args are additional arguments
        additional_args="$*"
        break
      fi
      ;;
  esac
done

if [ -z "$python_script" ]; then
  echo "Error: Python file to submit is required."
  print_usage
  exit 1
elif [ ! -f "$python_script" ]; then
  echo "Error: Python file '$python_script' does not exist."
  exit 1
fi

name=`basename $PWD`
name_flag="--job-name $name"
job_id=$(sbatch $name_flag $email_flag $ntasks_flag $ngpu_flag $queue_script $python_script $additional_args | awk '{print $4}')
echo "Submitted job $job_id to queue as $name"
echo $job_id
