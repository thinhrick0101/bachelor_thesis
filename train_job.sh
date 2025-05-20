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
#SBATCH --exclusive                       # Request exclusive access to nodes
#SBATCH --wait-all-nodes=1               # Wait for all nodes to be ready

# Enable error handling
set -e
set -x

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
export TORCH_DISTRIBUTED_TIMEOUT=1800  # 30 minutes timeout
export NCCL_DEBUG=INFO                 # Enable NCCL debugging
export NCCL_IB_TIMEOUT=30             # Increase InfiniBand timeout
export NCCL_SOCKET_TIMEOUT=300        # Increase socket timeout

# Print debug information
echo "=== Distributed Training Configuration ==="
echo "Master node: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "World size: $WORLD_SIZE"
echo "Job nodes: $SLURM_JOB_NODELIST"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "========================================"

# Ensure the logs directory exists
mkdir -p logs

# Launch with torchrun using srun
srun --kill-on-bad-exit=1 \
    torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=1 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --max_restarts=3 \
    stable_char_transformer.py

# Print final status
echo "Job completed with status $?"
