#!/bin/bash
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --job-name="prod_terra"
#SBATCH --array=1
#SBATCH --time=24:00:00
#SBATCH --output=/work/tesi_linghilterra/AIforBioinformatics/logs/slurm/slurm_out/%x_%j.out
#SBATCH --error=/work/tesi_linghilterra/AIforBioinformatics/logs/slurm/slurm_err/%x_%j.out
#SBATCH --cpus-per-task=8
#SBATCH --account=tesi_linghilterra
#SBATCH --constraint="gpu_1080_8G|gpu_2080Ti_11G|gpu_A40_48G|gpu_RTX5000_16G|gpu_RTX6000_24G|gpu_RTXA5000_24G"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=271539@studenti.unimore.it
#SBATCH --chdir=/work/tesi_linghilterra/AIforBioinformatics/src/binary_classification

module unload cuda/12.1
module load cuda/11.8

. /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate MachineLearning

# Imposta il PYTHONPATH per includere la directory principale del progetto
export PYTHONPATH=/work/tesi_linghilterra/AIforBioinformatics:$PYTHONPATH

# Esegui lo script Python
python /work/tesi_linghilterra/AIforBioinformatics/src/binary_classification/3.3_ge_\&_methylation_\&_os_gpu.py