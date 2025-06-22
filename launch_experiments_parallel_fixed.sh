#!/bin/bash
# launch_experiments_parallel_fixed.sh - Launch statistical experiments with fixed WandB auth

echo "Starting statistical analysis experiments on SLURM cluster..."
echo "This will submit 6 SLURM jobs (3 dense + 3 sparse) with different seeds"

# Configuration - SAME FOR BOTH MODELS (scientific validity)
SEEDS="111 222 333"
NUM_EPOCHS=50
BATCH_SIZE=16          # Same for both dense and sparse
SEQ_LENGTH=512         # Same for both dense and sparse  
LEARNING_RATE=1e-4

# Memory optimization: Use gradient accumulation instead o   different batch sizes
GRADIENT_ACCUMULATION_STEPS=2  # Effective batch size = 16 * 2 = 32

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

# Enhanced WandB setup with fallback
echo "🔑 Setting up WandB authentication..."

# Method 1: Try saved API key file
if [ -f ~/.wandb_api_key ]; then
    export WANDB_API_KEY=a0fcc743e67a1bed3ff2a929f609b2521ca3a154
    echo "✅ Using WandB API key from ~/.wandb_api_key"
# Method 2: Try environment variable  
elif [ ! -z "\$WANDB_API_KEY" ]; then
    echo "✅ Using WandB API key from environment"
# Method 3: Fallback to offline mode
else
    echo "⚠️ No WandB API key found - switching to offline mode"
    export WANDB_MODE=offline
fi

# Test WandB connection
if [ "\$WANDB_MODE" != "offline" ]; then
    echo "🧪 Testing WandB connection..."
    if ! python -c "import wandb; wandb.login(relogin=True)" 2>/dev/null; then
        echo "❌ WandB login failed - switching to offline mode"
        export WANDB_MODE=offline
    else
        echo "✅ WandB connection successful"
    fi
fi

export WANDB_CACHE_DIR=\$(pwd)/wandb_logs/.cache

# Clear CUDA cache before running
python -c "import torch; torch.cuda.empty_cache()"

# Run dense model training with error handling
echo "Starting dense model with seed ${seed}..."
if python -u train_dense_model.py \\
    --seed ${seed} \\
    --num_epochs ${NUM_EPOCHS} \\
    --wandb_run_name "dense_seed_${seed}" \\
    --batch_size ${BATCH_SIZE} \\
    --seq_length ${SEQ_LENGTH} \\
    --learning_rate ${LEARNING_RATE} \\
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS}; then
    echo "✅ Dense model seed ${seed} completed successfully"
else
    echo "❌ Dense model seed ${seed} failed with exit code \$?"
    exit 1
fi
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

# Enhanced WandB setup with fallback
echo "🔑 Setting up WandB authentication..."

# Method 1: Try saved API key file
if [ -f ~/.wandb_api_key ]; then
    export WANDB_API_KEY=a0fcc743e67a1bed3ff2a929f609b2521ca3a154
    echo "✅ Using WandB API key from ~/.wandb_api_key"
# Method 2: Try environment variable  
elif [ ! -z "\$WANDB_API_KEY" ]; then
    echo "✅ Using WandB API key from environment"
# Method 3: Fallback to offline mode
else
    echo "⚠️ No WandB API key found - switching to offline mode"
    export WANDB_MODE=offline
fi

# Test WandB connection
if [ "\$WANDB_MODE" != "offline" ]; then
    echo "🧪 Testing WandB connection..."
    if ! python -c "import wandb; wandb.login(relogin=True)" 2>/dev/null; then
        echo "❌ WandB login failed - switching to offline mode"
        export WANDB_MODE=offline
    else
        echo "✅ WandB connection successful"
    fi
fi

export WANDB_CACHE_DIR=\$(pwd)/wandb_logs/.cache

# Clear CUDA cache before running
python -c "import torch; torch.cuda.empty_cache()"

# Set memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1

# Run sparse model training with memory optimization
echo "Starting sparse model with seed ${seed} (same hyperparams as dense, memory optimized)..."
if python -u train_sparse_transformer.py \\
    --seed ${seed} \\
    --num_epochs ${NUM_EPOCHS} \\
    --wandb_run_name "sparse_seed_${seed}" \\
    --batch_size ${BATCH_SIZE} \\
    --seq_length ${SEQ_LENGTH} \\
    --learning_rate ${LEARNING_RATE} \\
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS}; then
    echo "✅ Sparse model seed ${seed} completed successfully"
else
    echo "❌ Sparse model seed ${seed} failed with exit code \$?"
    exit 1
fi
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
echo "🔍 Monitor jobs with:"
echo "  squeue -u \$USER"
echo "  squeue -j ${job_ids[*]}"
echo ""
echo "📋 Check logs in:"
echo "  logs/dense_seed_*_*.out"
echo "  logs/sparse_seed_*_*.out"
echo ""
echo "📊 After all jobs complete, run:"
echo "  python bachelor_thesis/aggregate_wandb_table.py"
echo "to generate the statistical summary table for your thesis."
echo ""
echo "================================================================"
echo "🚀 QUICK START - FIXING WANDB AUTHENTICATION"
echo "================================================================"
echo "If you're seeing WandB authentication errors, run this first:"
echo ""
echo "  bash bachelor_thesis/fix_wandb_auth.sh"
echo ""
echo "This will:"
echo "✅ Re-login to WandB with force relogin"
echo "✅ Save your API key for SLURM jobs"
echo "✅ Set up proper environment variables"
echo "✅ Test the connection"
echo ""
echo "Then re-run this script to submit jobs with fixed authentication."
echo "================================================================" 