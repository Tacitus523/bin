# Submit the python script to the queue and gives it the name of the current folder in the queue
# Use the -s flag to keep this process running until it finishes and sync training data to wandb
# Use the -c flag to submit a python file with a configuration file
# Use the -p flag to specify the python file to run
# Use the -e flag to send an email when the job finishes or fails

EMAIL="lukas.petersen@kit.edu"

queue_script="/lustre/home/ka/ka_ipc/ka_he8978/bin/qpython_justus.sh"

print_usage() {
  echo "Usage: 'submit_python_file.sh' to run without wandb or 'submit_python_file.sh -s' to sync to wandb"
  echo "Usage: 'submit_python_file.sh -p python_file' to submit a python file"
  echo "Usage: 'submit_python_file.sh -c config_path' to submit a python file with a configuration file"
}

sync=false
config_path=""
email_flag=""
while getopts ':p:c:se' flag; do
  case $flag in
    p)
      python_script="$OPTARG"
      echo "Using Python file: $python_script";;
    s) sync=true ;;
    c)
      config_path="$OPTARG"
      echo "Using configs: $config_path";;
    e) email_flag="--mail-user=$EMAIL --mail-type=END,FAIL" ;;
    \?)
      echo "Invalid option: -$OPTARG"
      print_usage
      exit 1;;
    :)
      echo "Option -$OPTARG requires an argument."
      print_usage
      exit 1;;
    *) print_usage
       exit 1 ;;
  esac
done

if [ -z "$python_script" ]
then
  echo "ERROR: Python file not specified."
  print_usage
  exit 1
fi

if [ -z "$config_path" ]
then echo "INFO: Did not get a config_file"
fi

name=`basename $PWD`
name_flag="--job-name $name"
job_id=$(sbatch $name_flag $email_flag $queue_script $python_script $config_path| awk '{print $4}')
echo "Submitted job $job_id to queue as $name"
echo $job_id
echo `date`" $PWD" >> $HOME/checklist.txt

# # for wandb sync
# if $sync
# then
#     nohup sync_wandb.sh $job_id &
# fi
