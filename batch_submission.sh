#!/usr/bin/bash
if [[ -z $1 || -z $2 || -z $3 ]]
then
    echo `date`" - Missing mandatory arguments:  start, end, batch_size"
    echo `date`" - Usage: ./batch_submission start end batch_size [padding_size]."
    echo `date`" - Usage: start and end should be suffixes of repositories in the current folder"
    exit 1
fi

set -o errexit # (or set -e) cause batch script to exit immediately when a command fails.

start=$1
end=$2
batch_size=$3
padding_size=$((${#end}))
batch_file=batch.sh # Script with actions per batch, see /home/lpetersen/bin/batch.sh for an example

if [ ! -z $4 ] && [ $4 -lt $padding_size ]
then
    echo "Requested padding is $4, which is less than the required padding for the end $end of $padding_size.\n"
    echo "Therefore no padding will be performed.\n"
    padding_size=0
elif [ ! -z $4 ]
then
    padding_size=$4
fi

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

