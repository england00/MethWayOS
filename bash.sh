#!/bin/bash
# Managing CUDA original_modules
module unload cuda/12.1
module load cuda/11.8

# Sourcing Conda and Activating Environment
. /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate MachineLearning

# GB: original params were: --mem=10G --time 4:00:00
srun -Q -wailb-login-02 --immediate=10 --mem=8G --partition=all_serial --gres=gpu:1 --nodes=1 --time 1:00:00 --pty --account=tesi_linghilterra bash