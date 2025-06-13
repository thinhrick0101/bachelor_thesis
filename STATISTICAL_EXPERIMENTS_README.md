# Statistical Experiments for Thesis

This directory contains the implementation of the pragmatic recipe for running statistical experiments to compare dense and sparse transformer models with proper mean ± standard deviation analysis.

## Overview

The setup runs **6 experiments total**:
- 3 dense model runs (seeds: 111, 222, 333)
- 3 sparse model runs (seeds: 111, 222, 333)

Each run logs 4 key metrics to WandB:
- `final_val_loss`: Final validation loss
- `final_val_ppl`: Final validation perplexity  
- `tokens_per_sec`: Training speed (tokens/second)
- `peak_gpu_mem_MB`: Peak GPU memory usage (MB)

## Quick Start

### 1. Validate Setup
```bash
python bachelor_thesis/validate_setup.py
```
This checks dependencies, CUDA, WandB login, and file availability.

### 2. Launch Experiments

**Sequential (safer for limited GPU memory):**
```bash
./bachelor_thesis/launch_experiments.sh
```

**Parallel (faster if you have enough GPU memory):**
```bash
./bachelor_thesis/launch_experiments_parallel.sh
```

### 3. Generate Results Table
```bash
python bachelor_thesis/aggregate_wandb_table.py
```

This produces:
- `detailed_metrics.csv`: Full statistical breakdown
- `summary_metrics.csv`: Clean summary table
- `table_metrics.tex`: LaTeX formatted table for thesis

## Files Modified/Created

### Core Training Scripts (Modified)
- `train_dense_model.py`: Added seed handling, argument parsing, metric logging
- `train_sparse_transformer.py`: Added seed handling, argument parsing, metric logging

### New Scripts
- `validate_setup.py`: Pre-flight checks
- `launch_experiments.sh`: Sequential experiment launcher
- `launch_experiments_parallel.sh`: Parallel experiment launcher
- `aggregate_wandb_table.py`: Statistical analysis and table generation

## Configuration

### Default Parameters
- **Epochs**: 20 (reduced from 100 for faster statistical analysis)
- **Seeds**: 111, 222, 333
- **Batch Size**: 32
- **Sequence Length**: 1024
- **Learning Rate**: 1e-4

### Customization
Edit the configuration section in the launcher scripts:
```bash
SEEDS="111 222 333"
NUM_EPOCHS=20
BATCH_SIZE=32
SEQ_LENGTH=1024
LEARNING_RATE=1e-4
```

## Expected Runtime

**With GPU (RTX 2080 Ti class):**
- Per run: ~40 minutes (20 epochs)
- Sequential: ~4 hours total
- Parallel: ~40 minutes (if GPU memory sufficient)

**With CPU:**
- 10-50x slower than GPU (not recommended)

## WandB Projects

Results are logged to separate WandB projects:
- `dense-transformer-training`: Dense model runs
- `sparse-transformer-training`: Sparse model runs

## Output Example

After running `aggregate_wandb_table.py`:

```
SUMMARY TABLE FOR THESIS
================================================
   Model  N        Loss    Perplexity Speed (tok/s)  Memory (MB)
   Dense  3  2.345 ± 0.023  10.4 ± 0.3   8543 ± 120    2847.3 ± 45.2
  Sparse  3  2.312 ± 0.031   10.1 ± 0.4   7892 ± 156    2654.1 ± 67.8
```

## Troubleshooting

### Common Issues

**1. WandB not logged in:**
```bash
wandb login
```

**2. CUDA out of memory (parallel mode):**
- Use sequential mode instead
- Reduce batch size in launcher scripts

**3. Missing dependencies:**
```bash
pip install torch numpy matplotlib wandb pandas
```

**4. Data not found:**
Data is downloaded automatically during first run.

### Logs

**Sequential mode:** Output goes to terminal
**Parallel mode:** Check individual log files:
- `dense_seed_111.log`, `dense_seed_222.log`, `dense_seed_333.log`  
- `sparse_seed_111.log`, `sparse_seed_222.log`, `sparse_seed_333.log`

## Validation Checklist

Before running experiments, ensure:

- ✅ Added `--seed` handling + logged it to wandb
- ✅ Logged **final_val_loss**, **final_val_ppl**, speed, memory
- ✅ Launched 3 seeds each for dense and sparse  
- ✅ All 6 runs show `state: finished` in WandB
- ✅ Aggregation script prints non-empty table

## Integration with Thesis

The generated LaTeX table (`table_metrics.tex`) can be directly included in your thesis:

```latex
\input{table_metrics.tex}
```

Or copy the summary statistics from the console output for manual formatting.

## Architecture Notes

- Both models use identical hyperparameters except for attention mechanism
- Reproducible results via fixed seeds
- Mixed precision training for efficiency
- Gradient accumulation for stability
- Early stopping to prevent overfitting 