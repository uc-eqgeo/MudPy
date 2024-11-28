#!/bin/bash -e
#SBATCH --job-name=FQ880-900 # job name (shows up in the queue)
#SBATCH --time=03:00:00      # Walltime (HH:MM:SS)
#SBATCH --mem-per-cpu=10GB
#SBATCH --ntasks 10
#SBATCH --nodes 1

#SBATCH --account=uc03610

#SBATCH -o logs/ruptures880-900_%j.out
#SBATCH -e logs/ruptures880-900_%j.err

# Activate the conda environment

mkdir -p logs

echo 'Purging modules and loading Miniconda'
module purge && module load Miniconda3

echo 'Sourcing conda'
source $(conda info --base)/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1

echo 'Activating mudypy environment'
conda activate mudpy

echo 'Preparing Python Path and MUD' 
export PYTHONPATH=$PYTHONPATH:/nesi/project/uc03610/jack/MudPy/src/python
export MUD=/nesi/project/uc03610/jack/MudPy/

echo 'Launching fakequakes'

python hikurangi_3D.fq.py 

# to call:
# sbatch slurm_example.sl
