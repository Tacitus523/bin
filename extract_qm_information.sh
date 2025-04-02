#Give folder-prefix as $1, file-prefix as $2
set -o errexit   # (or set -e) cause batch script to exit immediately when a command fails.

if [[ -z $1 || -z $2 ]]
then
    echo `date`" - Missing mandatory arguments: folder-prefix or file-prefix"
    echo `date`" - Usage: ./extract_qm_information.sh [folder-prefix] [file-prefix]"
    exit 1
fi

folders=$(find $1* -maxdepth 1 -type d | sort -V) # Ensures numerical ordering without padded folders --> folder_0, folder_1, folder_2, ... instead of folder_0, folder_1, folder_10, ... 
for folder in $folders
do
	if ! [ -f $folder/$2*.out ]
	then
		echo "No file named $folder/$2*.out"
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

FOLDER_FILE=folder_order.txt
CHARGES_MULL_FILE=charges_mull.txt
CHARGES_HIRSH_FILE=charges_hirsh.txt
CHARGES_LOEW_FILE=charges_loew.txt
#CHARGES_ESP_FILE=charges_esp.txt
ENERGIES_FILE=energies.txt
GEOMS_FILE=geoms.xyz
FORCES_FILE=forces.xyz
DIPOLE_FILE=dipoles.txt

remove_if_exists $FOLDER_FILE
remove_if_exists $CHARGES_MULL_FILE
remove_if_exists $CHARGES_HIRSH_FILE
remove_if_exists $CHARGES_LOEW_FILE
#remove_if_exists $CHARGES_ESP_FILE
remove_if_exists $ENERGIES_FILE
remove_if_exists $GEOMS_FILE
remove_if_exists $FORCES_FILE
remove_if_exists $DIPOLE_FILE

for folder in $folders
do
	num_atoms=$(grep -m 1 "Number of atoms" $folder/$2*.out | awk '{print $NF}')

	echo $folder >> $FOLDER_FILE
	# works with multip steps Orca .outs, only greps last occurence, tac = reverse cat, -m 1 = maximal 1 occurence 
	tac $folder/$2*.out | grep -B $(($num_atoms+1)) -m 1 'MULLIKEN ATOMIC CHARGES' | tac | awk 'FNR > 2 {print $4}' | tr '\n' ' ' >> $CHARGES_MULL_FILE
	tac $folder/$2*.out | grep -B $(($num_atoms+6)) -m 1 'HIRSHFELD ANALYSIS' | tac | awk 'FNR > 7 {print $3}' | tr '\n' ' ' >> $CHARGES_HIRSH_FILE
	tac $folder/$2*.out | grep -B $(($num_atoms+1)) -m 1 'LOEWDIN ATOMIC CHARGES' | tac | awk 'FNR > 2 {print $4}' | tr '\n' ' ' >> $CHARGES_LOEW_FILE
	#awk '{print $5}' $folder/$2.molden.chg | tr '\n' ' ' >> $CHARGES_ESP_FILE

	echo '' >> $CHARGES_MULL_FILE # basicially makes a \n
	echo '' >> $CHARGES_HIRSH_FILE # basicially makes a \n 
	echo '' >> $CHARGES_LOEW_FILE # basicially makes a \n
	#echo '' >> $CHARGES_ESP_FILE # basicially makes a \n 

	tac $folder/$2*.out | grep -m 1 'FINAL SINGLE' | tac | awk '{print $5}' >> $ENERGIES_FILE
	echo $num_atoms >> $GEOMS_FILE
	echo $folder >> $GEOMS_FILE
	tac $folder/$2*.out | grep -B $(($num_atoms+1)) -m 1 'CARTESIAN COORDINATES (ANGSTROEM)' | tac | awk 'FNR>2{print}' >> $GEOMS_FILE

	echo $num_atoms >> $FORCES_FILE
	echo $folder >> $FORCES_FILE
	tac $folder/$2*.out | grep -B $(($num_atoms+2)) -m 1 "CARTESIAN GRADIENT" | tac | awk 'FNR>3{printf "%s %+4.9f %+4.9f %+4.9f\n", $2, $4, $5, $6}' >> $FORCES_FILE

	tac $folder/$2*.out | grep -m 1 'Total Dipole Moment' | tac | awk '{print $5, $6, $7}' >> $DIPOLE_FILE
done
