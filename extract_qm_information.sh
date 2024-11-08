#Give atom number as $1, folder-prefix as $2, file-prefix as $3
set -o errexit   # (or set -e) cause batch script to exit immediately when a command fails.

if [[ -z $1 || -z $2 || -z $3 ]]
then
    echo `date`" - Missing mandatory arguments:  total atom number, folder-prefix or file-prefix"
    echo `date`" - Usage: ./extract_qm_information.sh  [total atom number] [folder-prefix] [file-prefix]"
    exit 1
fi

folders=$(find $2* -maxdepth 1 -type d | sort -V) # Ensures numerical ordering without padded folders --> folder_0, folder_1, folder_2, ... instead of folder_0, folder_1, folder_10, ... 
for folder in $folders
do
	if ! [ -f $folder/$3.out ]
	then
		echo "No file named $folder/$3.out"
		exit 1
	fi
	break
done

FOLDER_FILE=folder_order.txt
CHARGES_MULL_FILE=charges_mull.txt
CHARGES_HIRSH_FILE=charges_hirsh.txt
CHARGES_LOEW_FILE=charges_loew.txt
#CHARGES_ESP_FILE=charges_esp.txt
ENERGIES_FILE=energies.txt
GEOMS_FILE=geoms.xyz
FORCES_FILE=forces.xyz

if [ -f $FOLDER_FILE ]
then rm $FOLDER_FILE
fi

if [ -f $CHARGES_MULL_FILE ]
then rm $CHARGES_MULL_FILE
fi

if [ -f $CHARGES_HIRSH_FILE ]
then rm $CHARGES_HIRSH_FILE
fi

if [ -f $CHARGES_LOEW_FILE ]
then rm $CHARGES_LOEW_FILE
fi

# if [ -f $CHARGES_ESP_FILE ]
# then rm $CHARGES_ESP_FILE
# fi

if [ -f $ENERGIES_FILE ]
then rm $ENERGIES_FILE
fi

if [ -f $GEOMS_FILE ]
then rm $GEOMS_FILE
fi

if [ -f $FORCES_FILE ]
then rm $FORCES_FILE
fi

for folder in $folders
do
	echo $folder >> $FOLDER_FILE
	# works with multip steps Orca .outs, only greps last occurence, tac = reverse cat, -m 1 = maximal 1 occurence 
	tac $folder/$3.out | grep -B $(($1+1)) -m 1 'MULLIKEN ATOMIC CHARGES' | tac | awk 'FNR > 2 {print $4}' | tr '\n' ' ' >> $CHARGES_MULL_FILE
	tac $folder/$3.out | grep -B $(($1+6)) -m 1 'HIRSHFELD ANALYSIS' | tac | awk 'FNR > 7 {print $3}' | tr '\n' ' ' >> $CHARGES_HIRSH_FILE
	tac $folder/$3.out | grep -B $(($1+1)) -m 1 'LOEWDIN ATOMIC CHARGES' | tac | awk 'FNR > 2 {print $4}' | tr '\n' ' ' >> $CHARGES_LOEW_FILE
	#awk '{print $5}' $folder/$3.molden.chg | tr '\n' ' ' >> $CHARGES_ESP_FILE

	echo '' >> $CHARGES_MULL_FILE # basicially makes a \n
	echo '' >> $CHARGES_HIRSH_FILE # basicially makes a \n 
	echo '' >> $CHARGES_LOEW_FILE # basicially makes a \n
	#echo '' >> $CHARGES_ESP_FILE # basicially makes a \n 

	tac $folder/$3.out | grep -m 1 'FINAL SINGLE' | tac | awk '{print $5}' >> $ENERGIES_FILE
	echo $1 >> $GEOMS_FILE
	echo $folder >> $GEOMS_FILE
	tac $folder/$3.out | grep -B $(($1+1)) -m 1 'CARTESIAN COORDINATES (ANGSTROEM)' | tac | awk 'FNR>2{print}' >> $GEOMS_FILE

	echo $1 >> $FORCES_FILE
	echo $folder >> $FORCES_FILE
	tac $folder/$3.out | grep -B $(($1+2)) -m 1 "CARTESIAN GRADIENT" | tac | awk 'FNR>3{printf "%+4.9f %+4.9f %+4.9f\n", $4, $5, $6}' >> $FORCES_FILE
done
