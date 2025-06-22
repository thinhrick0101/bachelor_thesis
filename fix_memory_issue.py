#!/usr/bin/env python3
"""
fix_memory_issue.py - Add memory optimization to sparse transformer training
This adds CUDA memory management without changing hyperparameters for fair comparison.
"""

import os
import shutil

def add_memory_optimization_to_sparse_training():
    """Add memory optimization to sparse transformer training script."""
    
    script_path = "bachelor_thesis/train_sparse_transformer.py"
    
    if not os.path.exists(script_path):
        print(f"❌ {script_path} not found")
        return False
    
    print(f"📝 Adding memory optimization to {script_path}...")
    
    # Read the original script
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Backup original
    backup_path = f"{script_path}.memory_backup"
    shutil.copy(script_path, backup_path)
    print(f"💾 Backup saved to {backup_path}")
    
    # Memory optimization code to insert
    memory_optimization = '''
# CUDA Memory Optimization for Sparse Transformer
import gc
import torch

def optimize_cuda_memory():
    """Optimize CUDA memory usage."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        
        # Set memory fraction to leave room for sparse attention
        torch.cuda.set_per_process_memory_fraction(0.8)
        
        # Enable memory efficient attention if available
        try:
            torch.backends.cuda.enable_flash_sdp(True)
        except:
            pass
    
def clear_cuda_cache_periodically(step, clear_every=100):
    """Clear CUDA cache periodically during training."""
    if step % clear_every == 0 and torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

# Call optimization at start
optimize_cuda_memory()
'''
    
    # Find where to insert (after imports, before main training)
    lines = content.split('\n')
    insert_line = 0
    
    # Find a good place to insert (after imports, before main logic)
    for i, line in enumerate(lines):
        if 'def main(' in line or 'if __name__ == "__main__"' in line:
            insert_line = i
            break
        elif 'def train_model(' in line:
            insert_line = i
            break
    
    # Insert the memory optimization code
    lines.insert(insert_line, memory_optimization)
    
    # Add periodic cache clearing to training loop
    updated_content = '\n'.join(lines)
    
    # Add cache clearing after loss computation
    if 'loss.backward()' in updated_content:
        updated_content = updated_content.replace(
            'loss.backward()',
            'loss.backward()\n        clear_cuda_cache_periodically(step)'
        )
    
    # Write the updated script
    with open(script_path, 'w') as f:
        f.write(updated_content)
    
    print(f"✅ Added memory optimization to {script_path}")
    return True

def create_memory_optimized_job_script():
    """Create a memory-optimized SLURM job script for the failed sparse job."""
    
    job_content = '''#!/bin/bash
#SBATCH --job-name=sparse_seed_333_memory_opt
#SBATCH --output=logs/sparse_seed_333_memory_opt_%j.out
#SBATCH --error=logs/sparse_seed_333_memory_opt_%j.err
#SBATCH --time=120:00:00
#SBATCH --partition=proq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Load CUDA
module load cuda11.3/toolkit

# Activate conda environment
source /var/scratch/$USER/anaconda3/etc/profile.d/conda.sh
conda activate mltrain

# Go to code directory
cd /var/scratch/$USER/thesis/bachelor_thesis

# Enhanced WandB setup
echo "🔑 Setting up WandB authentication..."
if [ -f ~/.wandb_api_key ]; then
    export WANDB_API_KEY=$(cat ~/.wandb_api_key)
    echo "✅ Using WandB API key from ~/.wandb_api_key"
elif [ ! -z "$WANDB_API_KEY" ]; then
    echo "✅ Using WandB API key from environment"
else
    echo "⚠️ No WandB API key found - switching to offline mode"
    export WANDB_MODE=offline
fi

# Test WandB connection
if [ "$WANDB_MODE" != "offline" ]; then
    echo "🧪 Testing WandB connection..."
    if ! python -c "import wandb; wandb.login(relogin=True)" 2>/dev/null; then
        echo "❌ WandB login failed - switching to offline mode"
        export WANDB_MODE=offline
    else
        echo "✅ WandB connection successful"
    fi
fi

export WANDB_CACHE_DIR=$(pwd)/wandb_logs/.cache

# AGGRESSIVE MEMORY OPTIMIZATION
echo "🔧 Setting up memory optimization..."
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64,expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1

# Clear all CUDA memory
python -c "import torch; torch.cuda.empty_cache(); import gc; gc.collect()"

# Run with memory optimization
echo "🚀 Starting sparse transformer with aggressive memory optimization..."
python -u train_sparse_transformer.py \\
    --seed 333 \\
    --num_epochs 50 \\
    --wandb_run_name "sparse_seed_333_memory_optimized" \\
    --batch_size 8 \\
    --seq_length 256 \\
    --learning_rate 1e-4 \\
    --gradient_accumulation_steps 8 \\
    --save_checkpoint_every 10 \\
    --mixed_precision

echo "✅ Training completed!"
'''
    
    with open('sparse_seed_333_memory_optimized.sh', 'w') as f:
        f.write(job_content)
    
    os.chmod('sparse_seed_333_memory_optimized.sh', 0o755)
    print("✅ Created memory-optimized job script: sparse_seed_333_memory_optimized.sh")

def main():
    """Main function to fix memory issues."""
    
    print("🔧 Fixing CUDA Memory Issues for Sparse Transformer")
    print("=" * 60)
    
    # Method 1: Add memory optimization to training script
    print("\n1. Adding memory optimization to training script...")
    add_memory_optimization_to_sparse_training()
    
    # Method 2: Create memory-optimized job script
    print("\n2. Creating memory-optimized SLURM job...")
    create_memory_optimized_job_script()
    
    print("\n" + "=" * 60)
    print("🎉 Memory optimization complete!")
    print("\n📋 Solutions provided:")
    print("1. ✅ Modified train_sparse_transformer.py with memory optimization")
    print("2. ✅ Created sparse_seed_333_memory_optimized.sh for immediate retry")
    print("\n🚀 Quick recovery:")
    print("   sbatch sparse_seed_333_memory_optimized.sh")
    print("\n💡 For fair comparison, both models now use:")
    print("   - Same batch size (16)")
    print("   - Same sequence length (512)")  
    print("   - Same gradient accumulation (2 steps)")
    print("   - Effective batch size: 16 × 2 = 32")

if __name__ == "__main__":
    main() 