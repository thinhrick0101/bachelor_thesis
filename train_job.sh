#!/bin/bash
#SBATCH --job-name=dist_transformer      # Name of your job
#SBATCH --output=logs/%x_%j.out         # Save output to logs/jobname_jobid.out
#SBATCH --error=logs/%x_%j.err          # Save errors to logs/jobname_jobid.err
#SBATCH --time=24:00:00                 # Run time (hh:mm:ss)
#SBATCH --partition=defq                # Use defq partition
#SBATCH --nodes=4                       # Request 4 nodes
#SBATCH --ntasks-per-node=1            # One task per node
#SBATCH --gpus-per-node=1              # Request 1 GPU per node
#SBATCH --cpus-per-task=4              # 4 CPU cores per task
#SBATCH --mem=32G                      # Memory per node (DAS5 has 32GB per node)

# Load required modules
module load cuda11.3/toolkit
module load openmpi

# Activate conda environment
source /var/scratch/$USER/anaconda3/etc/profile.d/conda.sh
conda activate mltrain

# Go to code directory
cd /var/scratch/$USER/thesis/bachelor_thesis

# Create necessary directories
mkdir -p models logs

# Get the master node's hostname
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# Get the world size from SLURM
export WORLD_SIZE=$SLURM_NTASKS

# Create a simple sync script
cat > sync.py << 'EOL'
import torch.distributed as dist
import os
import sys

def init_process(rank, size, backend='nccl'):
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
    dist.init_process_group(backend, rank=rank, world_size=size)
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    rank = int(os.environ['SLURM_PROCID'])
    size = int(os.environ['SLURM_NTASKS'])
    init_process(rank, size)
EOL

# Step 1: Train the tokenizer first (only on master node)
if [ "$SLURM_PROCID" -eq 0 ]; then
    echo "=== Training BPE Tokenizer ==="
    python -u train_tokenizer.py \
        --data data/enwik8 \
        --vocab-size 8000 \
        --output models/tokenizer.json \
        --min-frequency 1
fi

# Wait for tokenizer to complete on all nodes using distributed synchronization
srun --ntasks=$SLURM_NTASKS --ntasks-per-node=1 python sync.py

# Check if tokenizer exists
if [ -f "models/tokenizer.json" ]; then
    echo "Tokenizer found. Starting distributed training..."
    
    # Step 2: Launch distributed training
    srun --ntasks=$SLURM_NTASKS --ntasks-per-node=1 \
        python -u stable_char_transformer.py \
        --local_rank=$SLURM_PROCID \
        --world_size=$WORLD_SIZE \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        --dist_backend="nccl"
else
    echo "Error: Tokenizer not found. Stopping job."
    exit 1
fi
