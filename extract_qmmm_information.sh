#Give folder-prefix as $1, file-prefix as $2
# Units
# ESP: eV/e
# ESP_gradient: eV/e/A
# MM_coordinate: A
# MM_charge: e
# MM_gradient: H/B (no confirmation)

ESPS_FILE=esps_by_qmmm.txt
PC_FILE=mm_data.pc
PCGRAD_FILE=mm_data.pcgrad

esp_calculation_script="esp_calculation_from_pc.py"

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

remove_if_exists() {
	local file=$1
	if [ -f $file ]
	then rm $file
	fi
}

extract_qm_information.sh $1 $2

remove_if_exists $ESPS_FILE
remove_if_exists $PC_FILE
remove_if_exists $PCGRAD_FILE

# # Concatenates esps from gromacs_dftb, if these are calculated with PME electrostatics, these include esps from QM zone
# for folder in $folders
# do
# 	tail -n 1 $folder/qm_dftb_esp.xvg | awk '{for (i=2; i<=NF; i++) print $i}' | tr '\n' ' ' >> $ESPS_FILE
# 	echo '' >> $ESPS_FILE # basicially makes a \n 
# done

for folder in $folders
do
	sed '/^$/d' $folder/$2.pc >> $PC_FILE # concatenate all pc files, remove empty lines
	sed '/^$/d' $folder/$2.pcgrad >> $PCGRAD_FILE # concatenate all pcgrad files, remove empty lines
done

remove_if_exists "esp_calc.out"
remove_if_exists "esp_calc.err"
qsub $(which $esp_calculation_script) --dir $1 --input $2 --unit V # ESP in Volt, change to au, if requiered

