#!/bin/bash

# set -o errexit   # (or set -e) cause batch script to exit immediately when a command fails.
# set -o pipefail  # cause batch script to exit immediately also when the command that failed is embedded in a pipeline
# set -o nounset   # (or set -u) causes the script to treat unset variables as an error and exit immediately 

# Give .inp as $1
input_file=$1
basename=${input_file%.in}
basename=${basename%.com}
basename=${basename%.inp}
output_file=$basename.out
job_file=slurm_orca.sh

if grep -iq "nprocs" $input_file
then
  cores=`grep -i nprocs $input_file | awk '{print $3}'` # Intented for 1-liner %PAL section, fails otherwise
  cores=$((cores + 1))
else
  cores=1
fi

cat > $job_file << EOF
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=6:00:00
#SBATCH --job-name=orca
#SBATCH --output=orca_job.out
#SBATCH --error=orca_job.err
#SBATCH --gres=scratch:100 # GB on scratch reserved

echo "JOB: \$SLURM_JOB_ID started on \$SLURM_CLUSTER_NAME -- \$date"
echo "Workdir: \$SCRATCH"

module load chem/orca/5.0.4

ulimit -s unlimited

current_dir=\$PWD
workdir="\$SCRATCH"
mkdir -p \$workdir

cp $input_file \$workdir
    
cd \$workdir  

\`which orca\` $input_file >> $output_file

# rm $basename.densities
# rm $basename.gbw

cd \$current_dir
rsync -a \$workdir/ .
rm -rf \$workdir

echo "Job done -- \$date"

EOF

sbatch $job_file