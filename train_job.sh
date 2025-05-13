#!/bin/bash
#SBATCH --job-name=char_transformer        # Name of your job
#SBATCH --output=logs/%x_%j.out           # Save output to logs/jobname_jobid.out
#SBATCH --error=logs/%x_%j.err            # Save errors to logs/jobname_jobid.err
#SBATCH --time=12:00:00                  # Run time (hh:mm:ss)
#SBATCH --partition=proq                  # Default queue (has GPUs)
#SBATCH --gres=gpu:1                      # Request 1 GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8                 # Adjust CPU cores if needed

# Load CUDA
module load cuda11.3/toolkit

# Activate your conda environment
source /var/scratch/$USER/anaconda3/etc/profile.d/conda.sh
conda activate mltrain

# Go to your code directory
cd /var/scratch/$USER/thesis/bachelor_thesis

# Run your training script (adjust paths/configs as needed)
python -u  stable_char_transformer.py
