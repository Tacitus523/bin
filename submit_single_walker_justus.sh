# Give the .tpr as $1 and the plumed as $2, and any other files as $3, $4, etc.

EMAIL=$MY_MAIL

WALKER_SCRIPT="single_walker_justus.sh"

print_usage() {
    echo "Usage: $0 -t TPR_FILE.tpr [-p PLUMED_FILE.dat] [-s simulation_type] [-e] [additional_files...]"
    echo "simulation_type can be 'pytorch', 'tensorflow', 'orca', or 'dftb'"
    exit 1
}

email_flag="--mail-user=$EMAIL --mail-type=END,FAIL" # Default email settings, always -e for now
simulation_type="pytorch"
while getopts ":t:p:s:e" opt
do
    case ${opt} in
        t )
            tpr_file=$OPTARG
            ;;
        p )
            plumed_file=$OPTARG
            ;;
        s )
            simulation_type=$OPTARG
            if [[ ! $simulation_type =~ ^(pytorch|pytorch_cpu|tensorflow|orca|dftb)$ ]]; then
                echo "Invalid simulation type: $simulation_type"
                print_usage
                exit 1
            fi
            ;;
        e )
            email_flag="--mail-user=$EMAIL --mail-type=END,FAIL"
            ;;
        \? )
            echo "Invalid option: -$OPTARG" 1>&2
            print_usage
            ;;
        : )
            echo "Invalid option: -$OPTARG requires an argument" 1>&2
            print_usage
            ;;
    esac
done
shift $((OPTIND - 1))

# Check if mandatory argument is set
if [[ ! $tpr_file == *.tpr ]]; then
    echo "tpr file must end with .tpr. Got $tpr_file"
    print_usage
    exit 1
fi
echo "tpr_file: $tpr_file"

if [ -z "$plumed_file" ]
then
    echo "Missing plumed file"
    plumed_flag=""
else
    echo "plumed_file: $plumed_file"
    plumed_flag="-p $plumed_file"
fi

# Remaining arguments are additional files
additional_files=("$@")
for file in "${additional_files[@]}"; do
    echo "Additional file: $file"
done

if [[ $simulation_type == "pytorch" ]]
then
    resource_flag="--gres=gpu:1"
    ntasks_flag="--ntasks-per-node=2" # 1 for gmx, 1 for observation script?
fi

if [[ $simulation_type == "pytorch_cpu" ]]
then
    ntasks_flag="--ntasks-per-node=2" # 1 for gmx, 1 for observation script?
fi

if [[ $simulation_type == "tensorflow" ]]
then
    resource_flag="" #"--gres=gpu:1"
    ntasks_flag="--ntasks-per-node=2" # 1 for gmx, 1 for observation script?
fi

if [[ $simulation_type == "orca" ]]
then
    # Check if *.ORCAINFO file is present in additional files
    found_orcainfo=false
    for file in "${additional_files[@]}"; do
        if [[ $file == *".ORCAINFO" ]]; then
            found_orcainfo=true
            export GMX_QM_ORCA_BASENAME=$(basename "$file" .ORCAINFO)
            echo "Using ORCAINFO file: $file. Basename set to: $GMX_QM_ORCA_BASENAME"
            break
        fi
    done
    if [ "$found_orcainfo" = false ]; then
        echo "ORCAINFO file is required for ORCA simulations."
        exit 1
    fi

    if grep -i "nprocs" $file > /dev/null
    then
    nprocs=`grep -i nprocs $file | awk '{print $3}'` # Only work with one-liner nprocs section
    else
    nprocs=1
    fi
    echo "Using $nprocs CPUs for ORCA simulation"
    ntasks_flag="--ntasks-per-node=$nprocs"
    resource_flag="--gres=scratch:250"
fi

if [[ $simulation_type == "dftb" ]]
then
    # Check if dftb_in.hsd file is present in additional files
    found_dftb_in=false
    for file in "${additional_files[@]}"; do
        if [[ $(basename "$file") == "dftb_in.hsd" ]]; then
            found_dftb_in=true
            break
        fi
    done
    if [ "$found_dftb_in" = false ]; then
        echo "WARNING: dftb_in.hsd file is required for DFTB simulations."
        #exit 1
    fi
fi

job_name=$(basename $tpr_file .tpr)_sim
echo "Job name: $job_name"
sbatch $email_flag $resource_flag $ntasks_flag --job-name=$job_name \
    $WALKER_SCRIPT -s $simulation_type -t $tpr_file $plumed_flag "${additional_files[@]}"
