if [ -z "$1" ]
  then
    file="/data/$USER/checklist.txt"
  else
    file="$1"
fi

echo "$file"

marker="-------------------------------"
train_file="train.err"

while read line; do
  #Skip Comments
  if [[ $line =~ ^# ]]; then continue; fi

  #Skip empty lines
  if [ -z $line ]; then continue; fi

  # Check for train-file existence
  if ! [ -f $line/$train_file ]
  then
    echo "Nothing trained in $line"
    continue
  fi

  # Check for succesful termination
  if ! grep -q "Terminated successfully" $line/$train_file 
  then
    echo "$line did not terminate successfully"
    continue
  fi

  # Print training results
  echo $line
  echo $marker
  grep "RMSE Charge:" $line/$train_file
  grep "R2 Charge:" $line/$train_file
  grep "RMSE Energy:" $line/$train_file
  grep "R2 Energy:" $line/$train_file
  echo $marker

done < "$file"