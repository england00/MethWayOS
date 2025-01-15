#!/bin/bash

#SBATCH --job-name=lucas_job
#SBATCH --mail-type=ALL
#SBATCH --mail-user=271539@studenti.unimore.it
#SBATCH --account=tesi_linghilterra

#SBATCH --partition=all_usr_prod
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --array=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_1080_8G|gpu_2080Ti_11G|gpu_A40_48G|gpu_RTX5000_16G|gpu_RTX6000_24G|gpu_RTXA5000_24G"

#SBATCH --output=/homes/linghilterra/AIforBioinformatics/logs/slurm/slurm_out/%x_%j.out
#SBATCH --error=/homes/linghilterra/AIforBioinformatics/logs/slurm/slurm_err/%x_%j.out
#SBATCH --chdir=/homes/linghilterra/AIforBioinformatics/src/mcat

# Choosing CUDA Module
module unload cuda/12.1
module load cuda/11.8

# Setting Conda Environment
. /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate MachineLearning
echo "Environment conda activated"

# Setting PYTHONPATH to include Root Directory of the project
echo "Job "$SLURM_JOB_ID" started"
export PYTHONPATH=/homes/linghilterra/AIforBioinformatics:$PYTHONPATH

# Run Python Script
python /homes/linghilterra/AIforBioinformatics/src/mcat/7_mcat_methylation_early_stopping.py