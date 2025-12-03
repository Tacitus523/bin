# Cut each walker in parallel using GNU parallel
# Usage: bash cut_traj.sh <input_xtc_file> <tpr_file> <idx_in_ndx_file> <ndx_file> <output_xtc_file>
# Example: bash cut_traj.sh traj_comp.xtc dipeptid_sol.tpr 1 idx.ndx cut_traj.xtc

module load system/parallel
module load lib/cudnn/9.0.0_cuda-12.3

set -o errexit   # (or set -e) cause batch script to exit immediately when a command fails.

PREFIX="WALKER_"
DEFAULT_IDX=1
DEFAULT_NDX_FILE=""
DEFAULT_OUTPUT_XTC="cut_traj.xtc"
DEBUG=false # Set to true to print all gromacs outputs

print_usage() {
    echo "Usage: $0 <input_xtc_file> <tpr_file> [idx_in_ndx_file] [ndx_file] [<output_xtc_file>]"
}

if [ -z "$1" ]
then
    echo "No input xtc file provided"
    print_usage
    exit 1
else
    input_xtc_file=$1
fi
if [ -z "$2" ]
then
    echo "No tpr file provided"
    print_usage
    exit 1
else
    tpr_file=$2
fi
if [ -z "$3" ]
then
    idx_in_ndx_file=$DEFAULT_IDX
else
    idx_in_ndx_file=$3
fi
if [ -z "$4" ]
then
    ndx_file=$DEFAULT_NDX_FILE
else
    ndx_file=$(realpath $4)
fi
if [ -z "$5" ]
then
    output_xtc_file=$DEFAULT_OUTPUT_XTC
else    
    output_xtc_file=$5
fi

n_walker=$(find . -type d -name "$PREFIX*" | wc -l)
echo "Number of walkers: $n_walker"
echo "input_xtc_file: $input_xtc_file"
echo "tpr file: $tpr_file"
echo "Index in ndx file: $idx_in_ndx_file"
echo "Output xtc file: $output_xtc_file"
echo "ndx file: $ndx_file"

cut_traj() {
    local walker_folder=$1
    local input_xtc_file=$2
    local tpr_file=$3
    local idx_in_ndx_file=$4
    local output_xtc_file=$5
    local ndx_file=$6 # Possibly empty, messes up the ordering of the arguments

    cd $walker_folder
    if [ -z "$ndx_file" ]
    then
        if $debug_cut_traj; then
            echo "$idx_in_ndx_file" | gmx -quiet trjconv -f $input_xtc_file -s $tpr_file -o $output_xtc_file
        else
            echo "$idx_in_ndx_file" | gmx -quiet trjconv -f $input_xtc_file -s $tpr_file -o $output_xtc_file > /dev/null 2>&1
        fi
    else
        if $debug_cut_traj; then
            echo "$idx_in_ndx_file" | gmx -quiet trjconv -f $input_xtc_file -s $tpr_file -n $ndx_file -o $output_xtc_file
        else
            echo "$idx_in_ndx_file" | gmx -quiet trjconv -f $input_xtc_file -s $tpr_file -n $ndx_file -o $output_xtc_file > /dev/null 2>&1
        fi
    fi
}

export -f cut_traj
export debug_cut_traj=$DEBUG
tpr_file=$(realpath $tpr_file)
parallel -j $n_walker "cut_traj {} $input_xtc_file $tpr_file $idx_in_ndx_file $output_xtc_file $ndx_file " ::: $PREFIX* &

wait