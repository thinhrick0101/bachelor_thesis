#!/bin/bash
# launch_experiments_parallel.sh - Launch statistical experiments on SLURM cluster

echo "Starting statistical analysis experiments on SLURM cluster..."
echo "This will submit 6 SLURM jobs (3 dense + 3 sparse) with different seeds"

# Configuration
SEEDS="111 222 333"
NUM_EPOCHS=50
BATCH_SIZE=32
SEQ_LENGTH=1024
LEARNING_RATE=1e-4

# Create necessary directories
mkdir -p bachelor_thesis/models
mkdir -p logs

# SLURM template for dense model
create_dense_job() {
    local seed=$1
    cat > "dense_seed_${seed}_job.sh" << EOF
#!/bin/bash
#SBATCH --job-name=dense_seed_${seed}       # Name of your job
#SBATCH --output=logs/dense_seed_${seed}_%j.out  # Save output
#SBATCH --error=logs/dense_seed_${seed}_%j.err   # Save errors
#SBATCH --time=120:00:00                    # Run time (hh:mm:ss)
#SBATCH --partition=proq                    # Default queue (has GPUs)
#SBATCH --gres=gpu:1                        # Request 1 GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8                   # Adjust CPU cores if needed

# Load CUDA
module load cuda11.3/toolkit

# Activate your conda environment
source /var/scratch/\$USER/anaconda3/etc/profile.d/conda.sh
conda activate mltrain

# Go to your code directory
cd /var/scratch/\$USER/thesis/bachelor_thesis

# Set up Weights & Biases for remote execution
export WANDB_API_KEY="YOUR_API_KEY"  # <-- IMPORTANT: REPLACE WITH YOUR KEY
export WANDB_CACHE_DIR=\$(pwd)/wandb_logs/.cache  # Store cache in project dir

# Uncomment the following line if your server has no internet access
# export WANDB_MODE=offline

# Clear CUDA cache before running
python -c "import torch; torch.cuda.empty_cache()"

# Run dense model training
echo "Starting dense model with seed ${seed}..."
python -u train_dense_model.py \\
    --seed ${seed} \\
    --num_epochs ${NUM_EPOCHS} \\
    --wandb_run_name "dense_seed_${seed}" \\
    --batch_size ${BATCH_SIZE} \\
    --seq_length ${SEQ_LENGTH} \\
    --learning_rate ${LEARNING_RATE}

echo "Dense model seed ${seed} completed"
EOF
    chmod +x "dense_seed_${seed}_job.sh"
}

# SLURM template for sparse model
create_sparse_job() {
    local seed=$1
    cat > "sparse_seed_${seed}_job.sh" << EOF
#!/bin/bash
#SBATCH --job-name=sparse_seed_${seed}      # Name of your job
#SBATCH --output=logs/sparse_seed_${seed}_%j.out # Save output
#SBATCH --error=logs/sparse_seed_${seed}_%j.err  # Save errors
#SBATCH --time=120:00:00                    # Run time (hh:mm:ss)
#SBATCH --partition=proq                    # Default queue (has GPUs)
#SBATCH --gres=gpu:1                        # Request 1 GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8                   # Adjust CPU cores if needed

# Load CUDA
module load cuda11.3/toolkit

# Activate your conda environment
source /var/scratch/\$USER/anaconda3/etc/profile.d/conda.sh
conda activate mltrain

# Go to your code directory
cd /var/scratch/\$USER/thesis/bachelor_thesis

# Set up Weights & Biases for remote execution
export WANDB_API_KEY="YOUR_API_KEY"  # <-- IMPORTANT: REPLACE WITH YOUR KEY
export WANDB_CACHE_DIR=\$(pwd)/wandb_logs/.cache  # Store cache in project dir

# Uncomment the following line if your server has no internet access
# export WANDB_MODE=offline

# Clear CUDA cache before running
python -c "import torch; torch.cuda.empty_cache()"

# Run sparse model training
echo "Starting sparse model with seed ${seed}..."
python -u train_sparse_transformer.py \\
    --seed ${seed} \\
    --num_epochs ${NUM_EPOCHS} \\
    --wandb_run_name "sparse_seed_${seed}" \\
    --batch_size ${BATCH_SIZE} \\
    --seq_length ${SEQ_LENGTH} \\
    --learning_rate ${LEARNING_RATE}

echo "Sparse model seed ${seed} completed"
EOF
    chmod +x "sparse_seed_${seed}_job.sh"
}

echo "================================================================"
echo "CREATING AND SUBMITTING SLURM JOBS"
echo "================================================================"

# Create and submit jobs for each seed
job_ids=()

for seed in $SEEDS; do
    echo "Creating job scripts for seed $seed..."
    
    # Create dense model job
    create_dense_job $seed
    echo "Submitting dense model job for seed $seed..."
    dense_job_id=$(sbatch "dense_seed_${seed}_job.sh" | awk '{print $4}')
    job_ids+=($dense_job_id)
    echo "  Dense job ID: $dense_job_id"
    
    # Create sparse model job  
    create_sparse_job $seed
    echo "Submitting sparse model job for seed $seed..."
    sparse_job_id=$(sbatch "sparse_seed_${seed}_job.sh" | awk '{print $4}')
    job_ids+=($sparse_job_id)
    echo "  Sparse job ID: $sparse_job_id"
    
    echo "----------------------------------------"
done

echo "================================================================"
echo "ALL JOBS SUBMITTED"
echo "================================================================"
echo "Submitted ${#job_ids[@]} jobs with IDs: ${job_ids[*]}"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo "  squeue -j ${job_ids[*]}"
echo ""
echo "Check logs in:"
echo "  logs/dense_seed_*_*.out"
echo "  logs/sparse_seed_*_*.out"
echo ""
echo "After all jobs complete, run:"
echo "  python bachelor_thesis/aggregate_wandb_table.py"
echo "to generate the statistical summary table for your thesis."
echo ""
echo "================================================================"
echo "IMPORTANT SETUP NOTES"
echo "================================================================"
echo "1. REPLACE 'YOUR_API_KEY' in the job scripts with your actual WandB API key"
echo "2. Adjust paths in job scripts if your setup differs from /var/scratch/\$USER/thesis/"
echo "3. Modify partition name if different from 'proq'"
echo "4. Check that cuda11.3/toolkit module is available (module avail cuda)"
echo "5. Ensure conda environment 'mltrain' exists and has required packages" 