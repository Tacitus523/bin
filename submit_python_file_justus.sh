# Submit the python script to the queue and gives it the name of the current folder in the queue
# Use the -s flag to keep this process running until it finishes and sync training data to wandb

<<<<<<< HEAD
python_script="/lustre/home/ka/ka_ipc/ka_he8978/kgcnn_fork/force_hdnnp4th.py"
queue_script="/lustre/home/ka/ka_ipc/ka_he8978/bin/qpython_justus.sh"
# For data readin in kgcnn
export BABEL_DATADIR="/opt/bwhpc/common/chem/openbabel/3.1.1"

print_usage() {
  echo "Usage: 'submit_python_file.sh' to run without wandb or 'submit_python_file.sh -s' to sync to wandb"
  echo "Usage: 'submit_python_file.sh -c config_path' to submit a python file with a configuration file"
}

sync=false
config_path=""
while getopts ':c:s' flag; do
  case $flag in
    s) sync=true ;;
    c)
      config_path="$OPTARG"
      echo "Using configs: $config_path";;
    \?)
      echo "Invalid option: -$OPTARG"
      print_usage
      exit 1;;
    :)
      echo "Option -$OPTARG requires an argument."
      print_usage
      exit 1;;
=======
#python_script="/home/lpetersen/dftb-nn/Behler_4th_Gen.py"
#python_script="/home/lpetersen/kgcnn_fork/hdnnp4th.py"
#python_script="/home/lpetersen/kgcnn_fork/force_hdnnp4th.py"
#python_script="/home/lpetersen/kgcnn_fork/hdnnp2nd.py"
python_script="/home/lpetersen/kgcnn_fork/force_hdnnp4th_hyp_param_search.py"
#python_script="/home/lpetersen/kgcnn_fork/charge_hyp_param_search.py"
#python_script="/home/lpetersen/kgcnn_fork/test_load_model.py"
#python_script="/home/lpetersen/kgcnn_fork/retrieve_trial.py"

queue_script="/home/lpetersen/bin/qpython.sh"

print_usage() {
  echo "Usage: 'submit_python_file.sh' to run without wandb or 'submit_python_file.sh -s' to sync to wandb"
}

sync=false
while getopts 's' flag; do
  case "${flag}" in
    s) sync=true ;;
>>>>>>> 9ab32ee2eed1d0a12c9db1b531265e70c5846fdd
    *) print_usage
       exit 1 ;;
  esac
done

<<<<<<< HEAD
if [ -z "$config_path" ]
then echo "Did not get a config_file"
fi

name=`basename $PWD`
job_id=$(sbatch --job-name $name $queue_script $python_script $config_path| awk '{print $4}')
echo "Submitted job $job_id to queue as $name"

echo `date`" $PWD" >> $HOME/checklist.txt

# # for wandb sync
# if $sync
# then
#     nohup sync_wandb.sh $job_id &
# fi
=======
# if [ -f train.err ]
# then rm train.err
# fi

# if [ -f train.out ]
# then rm train.out
# fi

name=`basename $PWD`
job_id=$(qsub -terse -N $name $queue_script $python_script)
echo "Submitted job $job_id to queue as $name"

echo `date`" $PWD" >> /data/$USER/checklist.txt

# for wandb sync
if $sync
then
    nohup sync_wandb.sh $job_id &
fi
>>>>>>> 9ab32ee2eed1d0a12c9db1b531265e70c5846fdd
