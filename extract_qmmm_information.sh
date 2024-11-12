#Give folder-prefix as $1, file-prefix as $2
set -o errexit   # (or set -e) cause batch script to exit immediately when a command fails.

if [[ -z $1 || -z $2 ]]
then
    echo `date`" - Missing mandatory arguments:  folder-prefix or file-prefix"
    echo `date`" - Usage: ./extract_qmmm_information.sh [folder-prefix]] [file-prefix] . "
    exit 1
fi

folders=$(find $1* -maxdepth 1 -type d | sort -V) # Ensures numerical ordering without padded folders --> folder_0, folder_1, folder_2, ... instead of folder_0, folder_1, folder_10, ... 
for folder in $folders
do
	if ! [ -f $folder/$2.out ]
	then
		echo "No file named $folder/$2.out"
		exit 1
	fi
	break
done

ESPS_FILE=esps_by_qmmm.txt

bash extract_qm_information.sh $1 $2

if [ -f $ESPS_FILE ]
then rm $ESPS_FILE
fi

# # Concatenates esps from gromacs_dftb, if these are calculated with PME electrostatics, these include esps from QM zone
# for folder in $1*
# do
# 	tail -n 1 $folder/qm_dftb_esp.xvg | awk '{for (i=2; i<=NF; i++) print $i}' | tr '\n' ' ' >> $ESPS_FILE
# 	echo '' >> $ESPS_FILE # basicially makes a \n 
# done

bash run_esp_calc.sh $1 $2
#qsub run_esp_calc.sh $1 $2
