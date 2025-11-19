#Give folder-prefix as $1, file-prefix as $2
# Units
# Geometries: Angstrom
# Energies: Hartree
# Gradients: Hartree/Bohr
# Charges: e
# Dipoles: e*Bohr (atomic units)
# Quadrupoles: e*Bohr^2 (atomic units)

set -o errexit   # (or set -e) cause batch script to exit immediately when a command fails.

if [[ -z $1 || -z $2 ]]
then
    echo `date`" - Missing mandatory arguments: folder-prefix or file-prefix"
    echo `date`" - Usage: ./extract_qm_information.sh [folder-prefix] [file-prefix]"
    exit 1
fi

folder_prefix=$1
# cut .out suffix from file_prefix
file_prefix=${2%.out}
folder_prefix_dirname=$(dirname "$folder_prefix")
folder_prefix_basename=$(basename "$folder_prefix")

folders=$(find "$folder_prefix_dirname" -maxdepth 1 -name "${folder_prefix_basename}*" \( -type d -o -type l \) | sort -V) # Ensures numerical ordering without padded folders --> folder_0, folder_1, folder_2, ... instead of folder_0, folder_1, folder_10, ... 
for folder in $folders
do
	if ! [ -f $folder/$file_prefix*.out ]
	then
		echo "No file named $folder/$file_prefix*.out"
		exit 1
	fi
	break
done
echo "Found $(echo $folders | wc -w) folders with prefix $folder_prefix"

remove_if_exists() {
	local file=$1
	if [ -f $file ]
	then rm $file
	fi
}

cat_if_exists() {
	local file=$1
	local target=$2
	if [ -f $file ]
	then cat $file >> $target
	fi
}

extract_multiwfn_charge() {
	local file=$1
	output_file=$2

	if [ -f $file ]
	then
		awk '{print $5}' $file | tr '\n' ' ' >> $output_file
		echo '' >> $output_file # basicially makes a \n
	fi
}

FOLDER_FILE=folder_order.txt
CHARGES_MULL_FILE=charges_mull.txt
CHARGES_HIRSH_FILE=charges_hirsh.txt
CHARGES_LOEW_FILE=charges_loew.txt
CHARGES_CHELPG_FILE=charges_chelpg.txt
CHARGES_MK_FILE=charges_mk.txt
CHARGES_EEM_FILE=charges_eem.txt
CHARGES_RESP_FILE=charges_resp.txt
CHARGES_MBIS_FILE=charges_mbis.txt
ENERGIES_FILE=energies.txt
GEOMS_FILE=geoms.xyz
GRADIENTS_FILE=gradients.xyz
DIPOLE_FILE=dipoles.txt
QUADRUPOLE_FILE=quadrupoles.txt
MULTIWFN_INPUTS=multiwfn_esp_input.pc
ESPS_BY_QM_FILE=esps_by_qm.pc

remove_if_exists $FOLDER_FILE
remove_if_exists $CHARGES_MULL_FILE
remove_if_exists $CHARGES_HIRSH_FILE
remove_if_exists $CHARGES_LOEW_FILE
remove_if_exists $CHARGES_CHELPG_FILE
remove_if_exists $CHARGES_MK_FILE
remove_if_exists $CHARGES_EEM_FILE
remove_if_exists $CHARGES_RESP_FILE
remove_if_exists $CHARGES_MBIS_FILE
remove_if_exists $ENERGIES_FILE
remove_if_exists $GEOMS_FILE
remove_if_exists $GRADIENTS_FILE
remove_if_exists $DIPOLE_FILE
remove_if_exists $QUADRUPOLE_FILE
remove_if_exists "forces.xyz" # Previously used file, renamed to gradients.xyz
remove_if_exists "charges_esp.txt" # Previously used file, renamed to charges_chelpg.txt
remove_if_exists $MULTIWFN_INPUTS
remove_if_exists $ESPS_BY_QM_FILE

counter=0
for folder in $folders
do
	is_converged="True"
	if ! tac $folder/$file_prefix*.out 2> /dev/null | grep -q -m 1 "****ORCA TERMINATED NORMALLY****" 2> /dev/null
    then 
        is_converged="False"
    fi

	echo "$counter,$folder,$is_converged" >> $FOLDER_FILE	
	counter=$((counter + 1))

	if [ $is_converged == "False" ]
	then
		continue
	fi

	num_atoms=$(grep -m 1 "Number of atoms" $folder/$file_prefix*.out | awk '{print $NF}')

	# works with multip steps Orca .outs, only greps last occurence, tac = reverse cat, -m 1 = maximal 1 occurence 
	tac $folder/$file_prefix*.out | grep -B $(($num_atoms+1)) -m 1 'MULLIKEN ATOMIC CHARGES' | tac | awk 'FNR > 2 {print $4}' | tr '\n' ' ' >> $CHARGES_MULL_FILE
	tac $folder/$file_prefix*.out | grep -B $(($num_atoms+6)) -m 1 'HIRSHFELD ANALYSIS' | tac | awk 'FNR > 7 {print $3}' | tr '\n' ' ' >> $CHARGES_HIRSH_FILE
	tac $folder/$file_prefix*.out | grep -B $(($num_atoms+1)) -m 1 'LOEWDIN ATOMIC CHARGES' | tac | awk 'FNR > 2 {print $4}' | tr '\n' ' ' >> $CHARGES_LOEW_FILE

	echo '' >> $CHARGES_MULL_FILE # basicially makes a \n
	echo '' >> $CHARGES_HIRSH_FILE # basicially makes a \n 
	echo '' >> $CHARGES_LOEW_FILE # basicially makes a \n

	extract_multiwfn_charge "$folder/CHELPG.chg" $CHARGES_CHELPG_FILE
	extract_multiwfn_charge "$folder/MK.chg" $CHARGES_MK_FILE
	extract_multiwfn_charge "$folder/EEM.chg" $CHARGES_EEM_FILE
	extract_multiwfn_charge "$folder/RESP.chg" $CHARGES_RESP_FILE
	extract_multiwfn_charge "$folder/MBIS.chg" $CHARGES_MBIS_FILE

	tac $folder/$file_prefix*.out | grep -m 1 'FINAL SINGLE' | tac | awk '{print $5}' >> $ENERGIES_FILE
	echo $num_atoms >> $GEOMS_FILE
	echo $folder >> $GEOMS_FILE
	tac $folder/$file_prefix*.out | grep -B $(($num_atoms+1)) -m 1 'CARTESIAN COORDINATES (ANGSTROEM)' | tac | awk 'FNR>2{print}' >> $GEOMS_FILE

	echo $num_atoms >> $GRADIENTS_FILE
	echo $folder >> $GRADIENTS_FILE
	tac $folder/$file_prefix*.out | grep -B $(($num_atoms+2)) -m 1 "CARTESIAN GRADIENT" | tac | awk 'FNR>3{printf "%s %+4.9f %+4.9f %+4.9f\n", $2, $4, $5, $6}' >> $GRADIENTS_FILE

	tac $folder/$file_prefix*.out | grep -m 1 'Total Dipole Moment' | tac | awk '{print $5, $6, $7}' >> $DIPOLE_FILE

	if  grep -q -m 1 'QUADRUPOLE MOMENT (A.U.)' $folder/$file_prefix*.out 2>/dev/null ; then
		tac $folder/$file_prefix*.out | grep -m 1 -B7 'QUADRUPOLE MOMENT (A.U.)' | grep 'TOT' | tac | awk '{print $2, $3, $4, $5, $6, $7}' >> $QUADRUPOLE_FILE
	else
		echo "nan nan nan nan nan nan" >> $QUADRUPOLE_FILE  # placeholder if missing
	fi

	cat_if_exists $folder/$MULTIWFN_INPUTS $MULTIWFN_INPUTS
	cat_if_exists $folder/$ESPS_BY_QM_FILE $ESPS_BY_QM_FILE

	if ((counter % 1000 == 0)) 
	then
		echo "Processed $counter folders"
	fi
done
