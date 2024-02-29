# For synchronising offline runs of wandb, should be run in the folder where training takes/took place
# Pass a job id and keep the process running in the background, while the model is training, using "nohup sync_wandb.sh job_id &",
# to upload the model after the training

# !!! Requires an environment with wandb for syncing. Give your environment here and it should be activated automatically regardless of active environment
WANDB_ENV=kgcnn
MAX_HOURS=24

if [ -z "$1" ]
then
    echo "No job_id supplied, looking for offline runs in wandb-folder"
else
    job_id=$1
    echo "Kill this process if it doesn't die: $$"
    echo "Tracking job id $job_id"
fi

#Check if Job id exists in the history of sun grid engine by checking the exit status of qacct, if job id does not exist, keep waiting, break after MAX_HOURS
# Start time of infinite loop
start_time=$( date "+%s" )
while true
do
    # No need to wait, if no job id was given
    if [ -z "$job_id" ]
    then break
    fi

    qacct -j "$job_id" 1> /dev/null 2> /dev/null # for some reason &> /dev/null seems to always give a 0 exit status
    qacct_exit_status=$?
    if [ $qacct_exit_status -eq 0 ] 
    then break
    else 
        sleep 60
        step_time=$( date "+%s" )
        duration=$(( step_time - start_time ))
        hours=$(( (duration % 86400) / 3600 ))
        if [ $hours -gt $MAX_HOURS ]
        then 
            echo "Job completion took too long, aborting sync"
            exit 1
        fi
    fi
done

# For WandB:
if [ -d wandb ]
then
    .  ${CONDA_PREFIX%/envs*}/etc/profile.d/conda.sh # should find the conda.sh script regardless of activated environment
    conda activate $WANDB_ENV
    cd wandb
    for offline_folder in offline-*
    do
        if [ -d $offline_folder ]
        then
            wandb sync $offline_folder
            mv $offline_folder ${offline_folder#"offline-"}
        fi
    done
    cd ..
else
    echo "No wandb folder around. Exiting"
    exit 2
fi

exit 0