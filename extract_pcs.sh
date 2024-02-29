#Give folder-prefix as $1, file-prefix as $2
set -o errexit   # (or set -e) cause batch script to exit immediately when a command fails.

if [[ -z $1 || -z $2 ]]
then
    echo `date`" - Missing mandatory arguments:  total atom number, folder-prefix or file-prefix"
    echo `date`" - Usage: ./extract_pcs.sh folder-prefix file-prefix"
    exit 1
fi

PCS_FILE=pcs.txt

if [ -f $PCS_FILE ]
then rm $PCS_FILE
fi

for folder in $1*
do
    tail -n +2 $folder/$2.pc >> $PCS_FILE # Skips the leading number of atoms
done
