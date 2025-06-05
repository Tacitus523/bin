#Give folder-prefix as $1
# atomic units

set -o errexit   # (or set -e) cause batch script to exit immediately when a command fails.

if [ -z $1 ]
then
    echo `date`" - Missing mandatory arguments: folder-prefix"
    echo `date`" - Usage: ./extract_qm_information.sh folder-prefix"
    exit 1
fi

folders=$(find $1* -maxdepth 1 -type d | sort -V) # Ensures numerical ordering without padded folders --> folder_0, folder_1, folder_2, ... instead of folder_0, folder_1, folder_10, ... 
for folder in $folders
do
	if ! [ -f "$folder/hamsqr1.dat" ]
	then
		echo "No file named $folder/hamsqr1.dat"
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

FOLDER_FILE=folder_order_hamiltonian_diagonal.txt
HAMILTONIAN_DIAGONAL_FILE=hamiltonian_diagonal.txt

remove_if_exists $FOLDER_FILE
remove_if_exists $HAMILTONIAN_DIAGONAL_FILE

for folder in $folders
do
	echo $folder >> $FOLDER_FILE
	awk 'NR > 5 { printf "%s ", $(NR-5) }' "$folder/hamsqr1.dat" >> $HAMILTONIAN_DIAGONAL_FILE # Extract the Hamiltonian diagonal from the output file
	echo "" >> $HAMILTONIAN_DIAGONAL_FILE
done

echo `date`" - Extracted Hamiltonian diagonal from DFTB+ output files to $HAMILTONIAN_DIAGONAL_FILE"

# Get the correct oribtals from the diagonal. ! WARNING: HARDCODED ELEMENAL ORDER
extract_dftb_hamiltonian_diagonal.py $HAMILTONIAN_DIAGONAL_FILE 

echo `date`" - Extracted correct orbitals from the Hamiltonian diagonal to dftb_hamiltonian_diagonal.txt"
