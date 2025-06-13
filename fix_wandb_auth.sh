#!/bin/bash
# fix_wandb_auth.sh - Fix WandB authentication on SLURM cluster

echo "🔑 Fixing WandB Authentication for SLURM Jobs"
echo "============================================="

# 1. Check current WandB status
echo "📊 Current WandB status:"
wandb status

# 2. Re-login to WandB (force relogin)
echo ""
echo "🔐 Forcing WandB relogin..."
wandb login --relogin

# 3. Verify login worked
echo ""
echo "✅ Verifying WandB login:"
wandb status

# 4. Set environment variables for SLURM jobs
echo ""
echo "📝 Setting up environment for SLURM..."

# Get WandB API key
WANDB_API_KEY=$(python -c "import wandb; print(wandb.api.api_key)")
echo "export WANDB_API_KEY=$WANDB_API_KEY" >> ~/.bashrc

# Alternative: Save to file for SLURM jobs
echo $WANDB_API_KEY > ~/.wandb_api_key
chmod 600 ~/.wandb_api_key

echo ""
echo "🎉 WandB authentication setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Source your bashrc: source ~/.bashrc"
echo "2. Test with: wandb status"
echo "3. Re-submit your SLURM jobs"
echo ""
echo "💡 If issues persist, use --use_wandb=false in training scripts" 