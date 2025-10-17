# Submit the python script to the queue and gives it the name of the current folder in the queue
# Use the -s flag to keep this process running until it finishes and sync training data to wandb

default_python_script="/home/lpetersen/kgcnn_fork/calc_prediction_std.py"

queue_script="/home/lpetersen/bin/qpython.sh"

print_usage() {
  echo "Usage: $0 python_script [-s] [-- additional_flags_and_args]"
  echo "  python_script : Path to the python script to run (required)"
  echo "  -s               : Keep the process running to sync wandb data (optional)"
  echo "  -- additional flags and args : Additional flags and arguments to pass to the python script (optional)"
}

sync=false
additional_args=""

# Process all arguments to separate flags from script name and additional args
while [ $# -gt 0 ]; do
  case $1 in
    -s) sync=true ;;
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
        echo "Error: Multiple python scripts specified. Please provide only one."
        print_usage
        exit 1
      fi
      ;;
  esac
done

if [ -z "$python_script" ]; then
  python_script=$default_python_script
fi
if [ ! -f "$python_script" ]; then
  echo "Error: Python file '$python_script' does not exist."
  exit 1
fi
echo "Submitting python script: $python_script"

# if [ -f train.err ]
# then rm train.err
# fi

# if [ -f train.out ]
# then rm train.out
# fi

name=`basename $PWD`
job_id=$(qsub -terse -N $name $queue_script $python_script $additional_args)
echo "Submitted job $job_id to queue as $name"

echo `date`" $job_id $name" >> /data/$USER/checklist.txt

# for wandb sync
if $sync
then
    nohup sync_wandb.sh $job_id &
fi
