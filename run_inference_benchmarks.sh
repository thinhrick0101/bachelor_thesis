#!/bin/bash

# Inference Benchmarking Script
# Runs latency and throughput tests for dense and sparse models

set -e

echo "=== ByteTransformer Inference Benchmarking ==="
echo "This script will benchmark available models for latency and throughput"
echo

# Set environment
export CUDA_VISIBLE_DEVICES=0

# Install required packages if not available
echo "Checking dependencies..."
python -c "import pynvml" 2>/dev/null || {
    echo "Installing nvidia-ml-py3 for GPU monitoring..."
    pip install nvidia-ml-py3
}

# Find available model checkpoints
echo "Looking for model checkpoints..."
MODELS=()

# Check for common model files
for model in *.pt; do
    if [[ -f "$model" ]]; then
        MODELS+=("$model")
    fi
done

if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "No .pt model files found in current directory!"
    echo "Available files:"
    ls -la *.pt 2>/dev/null || echo "  No .pt files found"
    echo
    echo "Please ensure you have trained model checkpoints available."
    echo "Expected files: dense_char_transformer.pt, sparse_model.pt, etc."
    exit 1
fi

echo "Found ${#MODELS[@]} model(s):"
for model in "${MODELS[@]}"; do
    echo "  - $model"
done
echo

# Create results directory
mkdir -p benchmark_results
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_FILE="benchmark_results/inference_results_${TIMESTAMP}.csv"

# Write CSV header
echo "Model,Batch,Latency (ms),Throughput (tok/s),GPU Util (%),Peak Mem (MB)" > "$RESULTS_FILE"

echo "Results will be saved to: $RESULTS_FILE"
echo

# Run benchmarks
for model in "${MODELS[@]}"; do
    echo "=== Benchmarking $model ==="
    
    # Extract model name for display
    model_name=$(basename "$model" .pt)
    
    echo "Testing batch size 1 (latency)..."
    python benchmark_inference.py --ckpt "$model" --batch 1 --iters 200 | grep "^$model_name," >> "$RESULTS_FILE"
    
    echo "Testing batch size 8 (throughput)..."
    python benchmark_inference.py --ckpt "$model" --batch 8 --iters 200 | grep "^$model_name," >> "$RESULTS_FILE"
    
    echo "Completed $model"
    echo
done

echo "=== Benchmark Summary ==="
echo "Results saved to: $RESULTS_FILE"
echo
echo "Summary table:"
echo "| Model | Batch | Latency (ms) | Tok/s | GPU util % | Peak Mem (MB) |"
echo "|-------|-------|--------------|-------|------------|---------------|"

# Display results in table format
tail -n +2 "$RESULTS_FILE" | while IFS=, read -r model batch latency throughput gpu_util peak_mem; do
    printf "| %-15s | %-5s | %-12s | %-8s | %-10s | %-13s |\n" \
        "$model" "$batch" "$latency" "$throughput" "$gpu_util" "$peak_mem"
done

echo
echo "=== Usage in Thesis ==="
echo "Copy the table above into your thesis Section 6.4"
echo "- Latency (batch=1): Per-sequence time for interactive use"
echo "- Throughput (batch=8): Tokens/second for offline processing"
echo "- GPU util: Justifies claims about SM saturation"
echo "- Peak Mem: Useful for edge device vs server GPU arguments"
echo

echo "=== Hardware Info ==="
echo "Document this hardware configuration:"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
echo "Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)"
echo "CUDA: $(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "Python: $(python --version)"
echo

echo "Benchmarking complete!" 