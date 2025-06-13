#!/bin/bash
# cleanup_disk_space.sh - Safe cleanup script for cluster disk space

echo "================================================================"
echo "CLUSTER DISK SPACE CLEANUP"
echo "================================================================"
echo "This script will safely clean up common space-consuming files."
echo "Press Ctrl+C to abort at any time."
echo ""

# Function to get directory size
get_size() {
    if [ -d "$1" ]; then
        du -sh "$1" 2>/dev/null | cut -f1
    else
        echo "0B"
    fi
}

# Function to ask for confirmation
confirm() {
    read -p "$1 (y/N): " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]]
}

echo "Current disk usage:"
quota -u $USER 2>/dev/null || df -h $HOME

echo ""
echo "Starting cleanup process..."

# 1. Clean Conda cache
echo ""
echo "1. CONDA CACHE CLEANUP"
echo "----------------------------------------"
if command -v conda &> /dev/null; then
    conda_cache_size=$(get_size "$HOME/.conda")
    echo "Conda cache size: $conda_cache_size"
    if confirm "Clean conda cache?"; then
        conda clean --all -y
        echo "✓ Conda cache cleaned"
    fi
else
    echo "Conda not found, skipping..."
fi

# 2. Clean pip cache
echo ""
echo "2. PIP CACHE CLEANUP"
echo "----------------------------------------"
if command -v pip &> /dev/null; then
    pip_cache_size=$(get_size "$HOME/.cache/pip")
    echo "Pip cache size: $pip_cache_size"
    if confirm "Clean pip cache?"; then
        pip cache purge
        echo "✓ Pip cache cleaned"
    fi
else
    echo "Pip not found, skipping..."
fi

# 3. Clean PyTorch cache
echo ""
echo "3. PYTORCH CACHE CLEANUP"
echo "----------------------------------------"
torch_cache_size=$(get_size "$HOME/.cache/torch")
echo "PyTorch cache size: $torch_cache_size"
if confirm "Clean PyTorch hub cache?"; then
    python -c "import torch; torch.hub.clear_cache()" 2>/dev/null
    rm -rf "$HOME/.cache/torch" 2>/dev/null
    echo "✓ PyTorch cache cleaned"
fi

# 4. Clean WandB cache
echo ""
echo "4. WANDB CACHE CLEANUP"
echo "----------------------------------------"
wandb_cache_size=$(get_size "$HOME/.cache/wandb")
echo "WandB cache size: $wandb_cache_size"
if confirm "Clean WandB cache?"; then
    rm -rf "$HOME/.cache/wandb" 2>/dev/null
    echo "✓ WandB cache cleaned"
fi

# 5. Clean general cache
echo ""
echo "5. GENERAL CACHE CLEANUP"
echo "----------------------------------------"
general_cache_size=$(get_size "$HOME/.cache")
echo "General cache size: $general_cache_size"
if confirm "Clean other cache directories?"; then
    # Keep important caches, remove others
    find "$HOME/.cache" -type d -name "*" ! -name "wandb*" ! -name "torch*" ! -name "pip*" -exec rm -rf {} + 2>/dev/null
    echo "✓ General cache cleaned"
fi

# 6. Clean old model files (be careful here)
echo ""
echo "6. OLD MODEL FILES"
echo "----------------------------------------"
echo "Looking for large model files..."
find $HOME -name "*.pt" -size +100M 2>/dev/null | while read file; do
    size=$(ls -lh "$file" | awk '{print $5}')
    echo "$file ($size)"
done

if confirm "Review and potentially remove large model files manually?"; then
    echo "Large model files found above. Remove them manually if no longer needed."
    echo "Example: rm /path/to/old_model.pt"
fi

# 7. Clean temporary files
echo ""
echo "7. TEMPORARY FILES"
echo "----------------------------------------"
tmp_size=$(get_size "/tmp")
echo "System temp size: $tmp_size"
if confirm "Clean your temporary files?"; then
    rm -rf /tmp/$USER* 2>/dev/null
    rm -rf "$HOME/tmp/*" 2>/dev/null
    echo "✓ Temporary files cleaned"
fi

# 8. Git cleanup
echo ""
echo "8. GIT CLEANUP"
echo "----------------------------------------"
echo "Cleaning git repositories..."
find $HOME -name ".git" -type d 2>/dev/null | while read git_dir; do
    repo_dir=$(dirname "$git_dir")
    echo "Cleaning $repo_dir"
    cd "$repo_dir" && git gc --aggressive --prune=now 2>/dev/null
done

echo ""
echo "================================================================"
echo "CLEANUP COMPLETE"
echo "================================================================"
echo "Disk usage after cleanup:"
quota -u $USER 2>/dev/null || df -h $HOME

echo ""
echo "Additional manual cleanup suggestions:"
echo "1. Remove unused conda environments: conda env list"
echo "2. Clean old jupyter checkpoints: find . -name '.ipynb_checkpoints' -exec rm -rf {} +"
echo "3. Remove old log files: find . -name '*.log' -mtime +30 -delete"
echo "4. Archive old experiments: tar -czf old_experiments.tar.gz old_exp_dir/"
echo ""
echo "For the training experiments, ensure you have at least 10GB free space." 