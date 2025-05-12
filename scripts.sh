#!/bin/bash
#SBATCH --job-name=char_transformer
#SBATCH --time=120:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -C RTX2080Ti
#SBATCH -p proq
#SBATCH --gres=gpu:1


# Load GPU drivers
. /etc/bashrc
. /etc/profile.d/lmod.sh
module load cuda12.6/toolkit
module load cuDNN/cuda12.3/9.1.0.70

# Set up proper Anaconda initialization
source $HOME/.bashrc
# Make sure conda command is available
if ! command -v conda &> /dev/null; then
    echo "conda command not found. Trying to initialize from standard location..."
    if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/miniconda3/etc/profile.d/conda.sh"
    else
        echo "Could not find conda installation. Please make sure Anaconda/Miniconda is installed."
        exit 1
    fi
fi

# Activate the base environment
conda activate

# Install required packages in the conda environment
conda install -y numpy matplotlib
conda install -y pytorch torchvision torchaudio -c pytorch
conda install -y tqdm

# Set up scratch directory to avoid disk quota issues
SCRATCH_DIR="/var/scratch/$USER"
mkdir -p $SCRATCH_DIR/transformer_job
cd $SCRATCH_DIR/transformer_job

# Copy code and data to scratch space
cp $HOME/bachelor_thesis/stable_char_transformer.py ./
mkdir -p data
cp $HOME/bachelor_thesis/data/enwik8 ./data/ 2>/dev/null || echo "Data file not found, downloading..."

# Download data if not found
if [ ! -f ./data/enwik8 ]; then
    python -c "
import urllib.request
import gzip
import os

# Create data directory
os.makedirs('data', exist_ok=True)

# Download enwik8
url = 'https://codeberg.org/pbm/former/raw/branch/master/data/enwik8.gz'
print(f'Downloading data from {url}')
urllib.request.urlretrieve(url, 'data/enwik8.gz')

# Decompress .gz file
with gzip.open('data/enwik8.gz', 'rb') as f_in:
    with open('data/enwik8', 'wb') as f_out:
        f_out.write(f_in.read())

print('Data downloaded and extracted')
"
fi

# Run directly from scratch space
echo "Running transformer model from $(pwd)"

# Verify Anaconda and packages are correctly loaded
python -c "
import sys
import numpy as np
import torch
import matplotlib
import tqdm

print('Python version:', sys.version)
print('NumPy version:', np.__version__)
print('PyTorch version:', torch.__version__)
print('Matplotlib version:', matplotlib.__version__)
print('Tqdm version:', tqdm.__version__)

print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU device:', torch.cuda.get_device_name(0))
    print('CUDA version:', torch.version.cuda)
    print('GPU memory:', torch.cuda.get_device_properties(0).total_memory / (1024**3), 'GB')
"

# Run the actual experiment with correct path
python ./stable_char_transformer.py

# Copy results back to home directory (if quota allows)
mkdir -p $HOME/results
cp *.png $HOME/results/ 2>/dev/null || echo "Could not copy plot files due to quota"
cp *.pt $HOME/results/ 2>/dev/null || echo "Could not copy model files due to quota"

# Print the location of the results
echo "Results are stored in: $PWD"
echo "Plots (if quota allows) are copied to: $HOME/results/"
