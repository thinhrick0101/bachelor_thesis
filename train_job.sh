#!/bin/bash
#SBATCH --job-name=byte_transformer        # Name of your job
#SBATCH --output=logs/%x_%j.out           # Save output to logs/jobname_jobid.out
#SBATCH --error=logs/%x_%j.err            # Save errors to logs/jobname_jobid.err
#SBATCH --time=120:00:00                  # Run time (hh:mm:ss)
#SBATCH --partition=defq                  # Default queue (has GPUs)
#SBATCH --gres=gpu:1                      # Request 1 GPU
#SBATCH --nodes=16                        # Request 16 nodes
#SBATCH --ntasks-per-node=1              # One task per node
#SBATCH --cpus-per-task=8                # Adjust CPU cores if needed

# Load CUDA
module load cuda11.3/toolkit

# Activate your conda environment
source /var/scratch/$USER/anaconda3/etc/profile.d/conda.sh
conda activate mltrain

# Go to your code directory
cd /var/scratch/$USER/thesis/bachelor_thesis

# Clear CUDA cache before running
python -c "import torch; torch.cuda.empty_cache()"

# Set up distributed training environment variables
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_JOB_NUM_NODES * SLURM_NTASKS_PER_NODE))

# Launch with torchrun
torchrun --nnodes=16 \
         --nproc_per_node=1 \
         --rdzv_id=$SLURM_JOB_ID \
         --rdzv_backend=c10d \
         --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
         stable_char_transformer.py
