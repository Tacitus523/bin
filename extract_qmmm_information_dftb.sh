#Give folder-prefix as $1, file-prefix as $2
# Units
# ESP: eV/e
# ESP_gradient: eV/e/A
# MM_coordinate: A
# MM_charge: e
# MM_gradient: H/B (no confirmation)

DETAIL_FILE="detailed.out"
TMP_GEOM_FILE="geom_tmp.xyz"

PC_FILE=mm_data.pc
PCGRAD_FILE=mm_data.pcgrad

esp_calculation_script="esp_calculation_from_pc.py"

set -o errexit   # (or set -e) cause batch script to exit immediately when a command fails.
set -o nounset   # (or set -u) causes the script to treat unset variables as an error and exit immediately

if [[ -z $1 || -z $2 ]]
then
    echo `date`" - Missing mandatory arguments: folder-prefix or file-prefix"
    echo `date`" - Usage: $0 [folder-prefix] [file-prefix]"
    exit 1
fi

folder_prefix=$1
gro_file=$2.gro
xyz_file=$2.xyz

folders=$(find $folder_prefix* -maxdepth 1 \( -type d -o -type l \) | sort -V) # Ensures numerical ordering without padded folders --> folder_0, folder_1, folder_2, ... instead of folder_0, folder_1, folder_10, ... 
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

# Convert the .gro file to .xyz, only works if the QM-zone is the first part of the .gro file
# Deprecated since switching to single point recalculations with point charges
convert_to_xyz() {
	local gro_file=$1
	local num_atoms=$2
	local comment=$3
	local xyz_file_path=$4

	local tmp_gro_file=tmp.gro

	if [ $gro_file == $tmp_gro_file ]
	then
		echo "Error: Temporary file name is the same as the input file name"
		exit 1
	fi

	echo $comment > $tmp_gro_file
	echo $num_atoms >> $tmp_gro_file
	head -n $(($num_atoms + 2)) $gro_file | tail -n $num_atoms >> $tmp_gro_file
	tail -n 1 $gro_file >> $tmp_gro_file
	obabel -igro $tmp_gro_file -O $xyz_file_path > /dev/null 2>&1
	rm $tmp_gro_file
}

# # Do the conversion to .xyz to prepare for the qm information extraction
# for folder in $folders
# do
# 	num_atoms=$(count_atoms $folder/$DETAIL_FILE)

# 	comment=$(realpath $folder)
# 	convert_to_xyz $folder/$gro_file $num_atoms $comment $folder/$TMP_GEOM_FILE
# done

remove_if_exists "esp_calc.out"
remove_if_exists "esp_calc.err"
qsub $(which $esp_calculation_script) --dir $1 --input $2 --unit V --format dftb # ESP in Volt, change to au, if requiered

# Call the qm information extraction script
extract_qm_information_dftb.sh $folder_prefix $xyz_file

echo "Extracting point charges and gradients from the DFTB detailed.out files..."
remove_if_exists $PC_FILE
remove_if_exists $PCGRAD_FILE
counter=0
for folder in $folders
do
	n_mm_atoms=$(grep 'forces_ext_charges' $folder/results.tag | awk -F, '{print $2}')
	echo $n_mm_atoms >> $PC_FILE
	sed -e 's/^[ \t]*//' -e '/^$/d' $folder/field.dat | awk '{print $4, $1, $2, $3}' >> $PC_FILE # Remove leading spaces, empty lines, and reorder columns to match orca format (charge, x, y, z)
	echo $n_mm_atoms >> $PCGRAD_FILE
	grep -A $n_mm_atoms 'forces_ext_charges' $folder/results.tag | tail -n +2 >> $PCGRAD_FILE

	counter=$((counter + 1))
	if (( $counter % 1000 == 0 ))
	then
		echo "Processed $counter folders"
	fi
done
