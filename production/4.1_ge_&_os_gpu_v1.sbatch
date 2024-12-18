#!/bin/bash

#SBATCH --job-name="prod_terra"
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

#SBATCH --output=/work/tesi_linghilterra/AIforBioinformatics/logs/slurm/slurm_out/%x_%j.out
#SBATCH --error=/work/tesi_linghilterra/AIforBioinformatics/logs/slurm/slurm_err/%x_%j.out
#SBATCH --chdir=/work/tesi_linghilterra/AIforBioinformatics/src/binary_classification

# Choosing CUDA Module
module unload cuda/12.1
module load cuda/11.8

# Setting Conda Environment
. /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate MachineLearning

# Setting PYTHONPATH to include Root Directory of the project
export PYTHONPATH=/work/tesi_linghilterra/AIforBioinformatics:$PYTHONPATH

# Run Python Script
python /work/tesi_linghilterra/AIforBioinformatics/src/binary_classification/4.1_ge_\&_os_gpu_v1.py