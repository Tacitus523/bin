#Give folder-prefix as $1, .gro file as $2
#TODO: A lot of information is still missing, no qmmm information is extracted, the parts here should probably go to extract_qmmm_information_dftb.sh

TMP_GEOM_FILE="geom_tmp.xyz"

set -o errexit   # (or set -e) cause batch script to exit immediately when a command fails.
set -o nounset   # (or set -u) causes the script to treat unset variables as an error and exit immediately

if [[ -z $1 || -z $2 ]]
then
    echo `date`" - Missing mandatory arguments: folder-prefix or .gro file"
    echo `date`" - Usage: ./extract_qm_information.sh [folder-prefix] [.gro file]"
    exit 1
fi

folder_prefix=$1
gro_file=$2

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

# Convert the .gro file to .xyz, only works if the QM-zone is the first part of the .gro file
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

# Do the conversion to .xyz to prepare for the qm information extraction
for folder in $folders
do
	num_atoms=$(count_atoms $folder/$DETAIL_FILE)

	comment=$(realpath $folder)
	convert_to_xyz $folder/$gro_file $num_atoms $comment $folder/$TMP_GEOM_FILE
done

# Call the qm information extraction script
extract_qm_information_dftb.sh $folder_prefix $TMP_GEOM_FILE

