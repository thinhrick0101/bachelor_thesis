#!/bin/bash
# optimize_storage_for_training.sh - Optimize storage for statistical experiments

echo "================================================================"
echo "STORAGE OPTIMIZATION FOR TRAINING EXPERIMENTS"
echo "================================================================"

# Check current space
echo "Current disk usage:"
quota -u $USER 2>/dev/null || df -h $HOME

echo ""
echo "Optimizing storage for training runs..."

# 1. Clean caches first
echo "1. Cleaning caches..."
pip cache purge 2>/dev/null
conda clean --all -y 2>/dev/null

# 2. Set up efficient WandB logging
echo "2. Configuring WandB for minimal storage..."
export WANDB_CACHE_DIR="/tmp/wandb_cache_$USER"
mkdir -p "/tmp/wandb_cache_$USER"

# 3. Configure model saving strategy
echo "3. Setting up model storage strategy..."
mkdir -p models/checkpoints
mkdir -p models/final

# 4. Clean old training artifacts
echo "4. Cleaning old training artifacts..."
find . -name "*.log" -mtime +7 -delete 2>/dev/null
find . -name "*_job.sh" -delete 2>/dev/null
find . -name "slurm-*.out" -delete 2>/dev/null

# 5. Set PyTorch memory management
echo "5. Configuring PyTorch for efficient memory usage..."
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"

# 6. Check available space
echo ""
echo "================================================================"
echo "STORAGE STATUS AFTER OPTIMIZATION"
echo "================================================================"
quota -u $USER 2>/dev/null || df -h $HOME

available_space=$(df -BG $HOME | tail -1 | awk '{print $4}' | sed 's/G//')
echo ""
if [ "$available_space" -gt 3 ]; then
    echo "✅ Storage optimized! ${available_space}GB available."
    echo "   This is sufficient for training experiments."
    echo ""
    echo "Next steps:"
    echo "1. Set your WandB API key in the job scripts"
    echo "2. Run: ./bachelor_thesis/launch_experiments_parallel.sh"
else
    echo "⚠️  Only ${available_space}GB available."
    echo "   Consider additional cleanup before training."
fi

echo ""
echo "================================================================"
echo "TRAINING STORAGE TIPS"
echo "================================================================"
echo "• Models will be saved to models/ directory"
echo "• WandB cache redirected to /tmp (auto-cleaned on reboot)"
echo "• Only final models will be kept (not intermediate checkpoints)"
echo "• Log files will be cleaned after 7 days"
echo "• Each training run needs ~200-300MB disk space" 