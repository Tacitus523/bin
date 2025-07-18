#!/bin/bash
#Give folder-prefix as $1, .xyz file as $2
# Units Output:
# Charges: e
# Energies: Hartree
# Forces: Hartree/Bohr
# Geometries: Angstrom

DETAIL_FILE="detailed.out"
TMP_FORCES_FILE="forces_tmp.xyz"

FOLDER_FILE=folder_order.txt
CHARGES_MULL_FILE=charges_mull.txt
ENERGIES_FILE=energies.txt
GEOMS_FILE=geoms.xyz
FORCES_FILE=forces.xyz

set -o errexit   # (or set -e) cause batch script to exit immediately when a command fails.
set -o nounset   # (or set -u) causes the script to treat unset variables as an error and exit immediately

if [[ -z $1 || -z $2 ]]
then
    echo `date`" - Missing mandatory arguments: folder-prefix or .xyz file"
    echo `date`" - Usage: $0 [folder-prefix] [.xyz file]"
    exit 1
fi

folder_prefix=$1
xyz_file=$2

folders=$(find $folder_prefix* -maxdepth 1 -type d | sort -V) # Ensures numerical ordering without padded folders --> folder_0, folder_1, folder_2, ... instead of folder_0, folder_1, folder_10, ... 
for folder in $folders
do
	if ! [ -f $folder/$DETAIL_FILE ]
	then
		echo "No file named $folder/$DETAIL_FILE"
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

# Function to count the number of atoms in the output file
count_atoms() {
	local file=$1
    local count=-1 # -1 to account for the header
    local start_counting=false

    while IFS= read -r line; do
        if [[ $line == " Atomic gross charges (e)" ]]; then # Careful with the space at the beginning
            start_counting=true
            continue
        fi

        if [[ $start_counting == true ]]; then
            if [[ -z "$line" ]]; then
                break
            fi
            count=$((count + 1))
        fi
    done < "$file"

    echo $count
}

# Write a forces file in xyz format based on the .xyz file, converts forces to gradients
write_forces_xyz() {
	local detail_file=$1
	local xyz_file=$2
	local num_atoms=$3
	local forces_file=$4

	local tmp_geom_file="_tmp_geom.xyz"
	local tmp_forces_file="_tmp_forces.txt"

	if [ $detail_file == $tmp_geom_file ] || [ $detail_file == $tmp_forces_file ]
	then
		echo "Error: Temporary file name is the same as the input file name"
		exit 1
	fi

	# Grep the forces, inverse the sign and write them to a temporary file
	grep -A $(($num_atoms+0)) -m 1 "Total Forces" $detail_file | awk 'FNR>1{printf "%+4.9f %+4.9f %+4.9f\n", -$2, -$3, -$4}' > $tmp_forces_file
	tail -n $num_atoms $xyz_file > $tmp_geom_file

	head -n 2 $xyz_file > $forces_file
	awk 'NR==FNR{a[NR]=$1;next}{print a[FNR], $1, $2, $3}' $tmp_geom_file $tmp_forces_file >> $forces_file

	rm $tmp_geom_file
	rm $tmp_forces_file
}

remove_if_exists $FOLDER_FILE
remove_if_exists $CHARGES_MULL_FILE
remove_if_exists $ENERGIES_FILE
remove_if_exists $GEOMS_FILE
remove_if_exists $FORCES_FILE

counter=0
for folder in $folders
do
	num_atoms=$(count_atoms $folder/$DETAIL_FILE)

	echo $folder >> $FOLDER_FILE
	grep -A $(($num_atoms+1)) -m 1 'Atomic gross charges (e)' $folder/$DETAIL_FILE | awk 'FNR > 2 {print $2}' | tr '\n' ' ' >> $CHARGES_MULL_FILE

	echo '' >> $CHARGES_MULL_FILE # basicially makes a \n, because I just replaced the \n with a space

	grep -m 1 'Total energy:' $folder/$DETAIL_FILE | awk '{print $3}' >> $ENERGIES_FILE

	comment=$(realpath $folder)
	cat $folder/$xyz_file >> $GEOMS_FILE

	write_forces_xyz $folder/$DETAIL_FILE $folder/$xyz_file $num_atoms $folder/$TMP_FORCES_FILE
	cat $folder/$TMP_FORCES_FILE >> $FORCES_FILE

	counter=$((counter + 1))
	if ((counter % 1000 == 0)) 
	then
		echo "Processed $counter folders"
	fi
done
