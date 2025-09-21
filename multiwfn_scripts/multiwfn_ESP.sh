#!/usr/bin/env bash
# Units:
# sp.pc: Angstrom, e
# multiwfn input: Bohr, e
# output: Bohr, 


print_usage() {
    echo "Usage: $0 <input_molden_input>"
    echo "Example: $0 your_file.molden.input"
    echo "This script runs Multiwfn on the specified molden-input('orca_2mkl YOUR_FILE.gbw -molden') file."
}

input_file="$1"
input_base=$(basename "$input_file" .molden.input)
output_prefix="ESP_multiwfn"

script_path=$(readlink -f "$0")
script_folder=$(dirname "$script_path")
pc_conversion_script="$script_folder/convert_pc_file_to_multiwfn_input.py"
orca_pc_file="${input_base}.pc"
pc_conversion_out="multiwfn_esp_input.pc"
mm_subsample_size=100
esp_calculation_out="esps_by_qm.pc"
multiwfn_input="5\n12\n100\n$pc_conversion_out\n$esp_calculation_out\nq"

if [ -z "$input_file" ]; then
    print_usage
    exit 1
fi

if ! [ -f "$input_file" ]; then
    echo "Input file '$input_file' does not exist."
    exit 1
fi

if ! which Multiwfn > /dev/null; then
    echo "Multiwfn is not installed or not in PATH."
    exit 1
fi

$pc_conversion_script -f $orca_pc_file -n $mm_subsample_size -o $pc_conversion_out
echo -e "$multiwfn_input" | Multiwfn $input_file > ${output_prefix}.log

if [ $? -ne 0 ]; then
    echo "Multiwfn execution for $output_prefix failed."
    exit 1
fi
echo "Multiwfn execution for $output_prefix completed successfully."

# Dont remove, still need the subsampled MM charges for merging with ESPs
#rm $pc_conversion_out # Remove temporary pc file