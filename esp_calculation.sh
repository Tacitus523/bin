export PATH="/home/lpetersen/miniconda3/bin:$PATH"
source /home/lpetersen/miniconda3/etc/profile.d/conda.sh
conda activate venv_tf_new
source /home/tkubar/GMX-DFTB/gromacs-dftbplus/release-machine-learning/bin/GMXRC
python3 /home/lpetersen/bin/esp_calculation.py