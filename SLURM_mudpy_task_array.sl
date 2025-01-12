#!/bin/bash -e
#SBATCH --job-name=Fakequakes # job name (shows up in the queue)
#SBATCH --time=01:00:00      # Walltime (HH:MM:SS)
#SBATCH --mem=13GB
#SBATCH --array=0-15

#SBATCH --account=uc03610

#SBATCH -o logs_ta/ruptures%j_task%a.out
#SBATCH -e logs_ta/ruptures%j_task%a.err

# Activate the conda environment

mkdir -p logs_ta

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
task_array=task_arrays
cp ${task_array}.txt ${task_array}_running.txt

awk -v target_line=$((SLURM_ARRAY_TASK_ID+1)) 'NR == target_line {print $0}' ${task_array}.txt

python hikurangi_3D_task_array.fq.py --task_number $SLURM_ARRAY_TASK_ID --task_file ${task_array}.txt

# to call:
# sbatch slurm_example.sl
