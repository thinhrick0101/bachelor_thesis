#!/bin/bash
#SBATCH --job-name=test
#SBATCH --time=00:15:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=defq
#SBATCH -C TitanX
#SBATCH --gres=gpu:1

# Load GPU drivers
. /etc/bashrc
. /etc/profile.d/lmod.sh
module load cuda11.7/toolkit
module load cuDNN/cuda11.7

# This loads the anaconda virtual environment with our packages
source $HOME/.bashrc
conda activate

# Base directory for the experiment
mkdir -p $HOME/experiments
cd $HOME/experiments

# Simple trick to create a unique directory for each run of the script
echo $$
mkdir -p o`echo $$`
cd o`echo $$`

# Test if GPU is properly detected
python <<EOF
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
EOF 