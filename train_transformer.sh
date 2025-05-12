#!/bin/bash
#SBATCH --job-name=char_transformer
#SBATCH --time=72:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p proq
#SBATCH -C RTX2080Ti
#SBATCH --gres=gpu:1
#SBATCH --output=transformer_%j.out
#SBATCH --error=transformer_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tng204@vu.nl

# Load GPU drivers and environment modules
. /etc/bashrc
. /etc/profile.d/lmod.sh
module load cuda11.7/toolkit
module load cuDNN/cuda11.7

# This loads the anaconda environment with our packages
source $HOME/.bashrc
conda activate

# Create directories for output
EXPERIMENT_DIR=$HOME/experiments/char_transformer_$(date +%Y%m%d_%H%M%S)
mkdir -p $EXPERIMENT_DIR
mkdir -p $HOME/data
mkdir -p $EXPERIMENT_DIR/checkpoints

# Copy the script to the experiment directory for reproducibility
cp $HOME/bachelor_thesis/stable_char_transformer.py $EXPERIMENT_DIR/
cp $HOME/bachelor_thesis/run_transformer.py $EXPERIMENT_DIR/
cp $HOME/bachelor_thesis/checkpoint_utils.py $EXPERIMENT_DIR/

# Change to the experiment directory
cd $EXPERIMENT_DIR

# Create data directory if it doesn't exist
mkdir -p data

# Configure Python to save checkpoints
export PYTHONUNBUFFERED=1

# Run the training script
echo "Starting training at $(date)"
python run_transformer.py \
  --data_path $HOME/data/enwik8 \
  --checkpoint_dir $EXPERIMENT_DIR/checkpoints \
  --batch_size 64 \
  --seq_length 512 \
  --num_epochs 25 \
  --learning_rate 3e-4 \
  --checkpoint_freq 1 \
  --max_chars 3000000

# After training completes, copy results to a permanent location
echo "Training completed at $(date)"
mkdir -p $HOME/results/char_transformer_$(date +%Y%m%d)
cp *.png $HOME/results/char_transformer_$(date +%Y%m%d)/
cp *.pt $HOME/results/char_transformer_$(date +%Y%m%d)/
cp -r checkpoints $HOME/results/char_transformer_$(date +%Y%m%d)/

echo "Results saved to $HOME/results/char_transformer_$(date +%Y%m%d)/" 