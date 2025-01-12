#!/bin/bash -e
#SBATCH --job-name=pygmo # job name (shows up in the queue)
#SBATCH --time=01:00:00      # Walltime (HH:MM:SS)
#SBATCH --mem-per-cpu=2GB
#SBATCH --cpus-per-task=10

#SBATCH --account=uc03610

#SBATCH -o logs_pygmo/b11N215_%j.out
#SBATCH -e logs_pygmo/b11N215_%j.err

# Activate the conda environment

mkdir -p logs_pygmo

echo 'Purging modules and loading Miniconda'
module purge
module load Miniconda3

echo 'Sourcing conda'
source $(conda info --base)/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1

echo 'Activating pygmo'
conda activate pygmo

#env > tty.env
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}


export TMPDIR="$PWD"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}

echo 'Launching fakequakes inversion'
echo ''
echo '********************************************'
echo ''

python -u scripts/deficit_inversion_multi.py

# sbatch slurm_example.sl
