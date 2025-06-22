# Inference Benchmarking for ByteTransformer Models

This directory contains a complete benchmarking system for measuring inference performance of both dense and sparse ByteTransformer models. The system measures **latency**, **throughput**, **GPU utilization**, and **memory usage** to provide hard numbers for your thesis.

## Quick Start

### Option 1: Automated Benchmarking (Recommended)

Run all benchmarks automatically:

```bash
cd bachelor_thesis
./run_inference_benchmarks.sh
```

This will:
- Find all `.pt` model files in the current directory
- Run batch=1 tests (latency) and batch=8 tests (throughput) for each model
- Generate a summary table ready for your thesis
- Save results to `benchmark_results/inference_results_TIMESTAMP.csv`

### Option 2: Manual Benchmarking

For fine-grained control:

```bash
# Test sparse model latency (batch=1)
python benchmark_inference.py --ckpt sparse_model.pt --batch 1 --iters 200

# Test sparse model throughput (batch=8)  
python benchmark_inference.py --ckpt sparse_model.pt --batch 8 --iters 200

# Test dense model latency (batch=1)
python benchmark_inference.py --ckpt dense_char_transformer.pt --batch 1 --iters 200

# Test dense model throughput (batch=8)
python benchmark_inference.py --ckpt dense_char_transformer.pt --batch 8 --iters 200
```

## Files

- **`benchmark_inference.py`**: Main benchmarking script
- **`run_inference_benchmarks.sh`**: Automated runner for all models
- **`benchmark_results/`**: Directory containing timestamped results

## Requirements

```bash
pip install nvidia-ml-py3  # For GPU utilization monitoring
```

All other dependencies should already be available in your environment.

## Understanding the Output

The benchmarking system produces a table like this:

| Model | Batch | Latency (ms) ↓ | Tok/s ↑ | GPU util % | Peak Mem (MB) |
|-------|-------|----------------|---------|------------|---------------|
| dense | 1     | 15.2          | 67,368  | 45.3       | 1,248         |
| dense | 8     | 8.7           | 747,126 | 89.1       | 3,456         |
| sparse| 1     | 12.8          | 80,000  | 38.7       | 1,156         |
| sparse| 8     | 7.2           | 904,762 | 82.4       | 3,201         |

### Metrics Explained

- **Latency (ms)**: Time per sequence (batch=1). Critical for interactive applications.
- **Tok/s**: Tokens per second throughput (batch=8). Important for offline processing.
- **GPU util %**: Average GPU utilization during inference. Helps justify efficiency claims.
- **Peak Mem (MB)**: Maximum GPU memory used. Relevant for deployment constraints.

## Technical Details

### Timing Methodology
- Uses `torch.cuda.Event` pairs for precise GPU timing
- Includes 10-iteration warmup to eliminate kernel compilation overhead
- Synchronizes with `torch.cuda.synchronize()` for wall-clock accuracy
- Runs with `torch.inference_mode()` for production-like conditions

### GPU Monitoring
- Samples GPU utilization every 50ms in a separate process
- Uses NVIDIA Management Library (NVML) for accurate metrics
- Tracks both compute and memory utilization

### Data Generation
- Uses real enwik8 data when available (`data/enwik8`)
- Falls back to realistic dummy data for consistent benchmarking
- Maintains exact 1024-byte sequence length used in training

### Model Loading
- Automatically detects sparse vs dense models from filename
- Handles both direct state_dict and checkpoint formats
- Configures models with production settings (no gradient checkpointing)

## Advanced Usage

### Custom Configuration

```bash
# Use torch.compile for additional optimization
python benchmark_inference.py --ckpt model.pt --batch 1 --compile

# Custom sequence length and iteration count
python benchmark_inference.py --ckpt model.pt --batch 1 --seq_length 512 --iters 500

# Benchmark with different batch sizes
python benchmark_inference.py --ckpt model.pt --batch 16 --iters 100
```

### Troubleshooting

**Model loading errors**: The script tries to auto-detect model type. If this fails, check that your model files follow the expected naming convention (`sparse*` for sparse models).

**GPU monitoring disabled**: Install `nvidia-ml-py3` for full GPU utilization tracking.

**No enwik8 data found**: The script will use dummy data, which is fine for performance benchmarking.

## Integration with Thesis

### Section 6.4: Inference Performance

Add this subsection after your training results. Include:

1. **Table**: Copy the generated table directly
2. **Hardware specs**: Document GPU model, driver, CUDA, PyTorch versions
3. **Methodology**: Reference this benchmarking script in your reproducibility checklist

### Key Claims to Support

- **Latency advantage**: "Sparse attention reduces per-sequence latency by X% at batch=1"
- **Throughput scaling**: "Dense models achieve higher GPU utilization at batch=8"  
- **Memory efficiency**: "Sparse patterns reduce peak memory usage by Y MB"
- **Hardware utilization**: "GPU utilization patterns confirm SM saturation differences"

### Sample Text

> "Inference performance was evaluated using the same 1024-byte sequences from the enwik8 validation set. We measured both single-sequence latency (batch=1) for interactive applications and multi-sequence throughput (batch=8) for offline processing. All measurements used CUDA Events with synchronization barriers to ensure wall-clock accuracy, following the timing methodology established in the training analysis."

## Reproducibility

To ensure reproducible results:

1. Use the same GPU model as training
2. Set `CUDA_VISIBLE_DEVICES=0` for single-GPU benchmarking  
3. Run with the exact model checkpoints used for evaluation
4. Document all software versions (automatically printed by the runner script)

## Citation

Include this benchmarking methodology in your thesis appendix and cite the script in your reproducibility checklist. 