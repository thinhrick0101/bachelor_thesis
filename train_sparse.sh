#!/bin/bash

# Activate Python environment if needed
# source /path/to/your/env/bin/activate

# Set CUDA device if needed
export CUDA_VISIBLE_DEVICES=0

echo "Starting sparse transformer training..."

# Train the sparse transformer model
python train_sparse_transformer.py

echo "Training completed. Starting attention pattern analysis..."

# Analyze attention patterns
python analyze_sparse_attention.py

echo "Analysis completed. Results are saved in attention_analysis/sparse_comparison/" 