# W&B Metrics Extraction for Thesis

This directory contains scripts to extract metrics directly from your W&B runs and generate thesis-ready tables without re-training.

## Files Created

- `make_metrics_csv.py` - Core script to pull metrics from W&B API
- `generate_table.py` - Generate LaTeX and Markdown tables from CSV
- `create_metrics_table.py` - Convenience script that runs both steps
- `WANDB_METRICS_README.md` - This guide

## Quick Start

### Option 1: One-step solution (Recommended)
```bash
python create_metrics_table.py
```

### Option 2: Step-by-step
```bash
# Step 1: Extract metrics from W&B
python make_metrics_csv.py

# Step 2: Generate tables
python generate_table.py
```

## Prerequisites

```bash
pip install wandb pandas
wandb login  # paste your API key when prompted
```

## Configuration

The scripts are pre-configured with your run details:

**Projects:**
- Dense: `dense-transformer-training`
- Sparse: `sparse-transformer-training`

**Run IDs:**
- Dense: `dense_run_1749531571`
- Sparse: `run_1749328061`

**Metrics Extracted:**
- `loss` - Final validation loss
- `ppl` - Final validation perplexity  
- `speed` - Tokens per second
- `mem` - Peak GPU memory (MB)

## Output Files

After running, you'll get:

```
metrics.csv          # Raw CSV data
table_metrics.tex    # LaTeX table for thesis
table_metrics.md     # Markdown preview
```

## Using in Your Thesis

### LaTeX Integration

Include the generated table in your thesis:

```latex
\input{table_metrics.tex}
```

Or copy-paste the content from `table_metrics.tex` directly.

### Sample LaTeX Output

```latex
\begin{table}[htbp]
\centering
\caption{Performance comparison between dense and sparse transformer models. Lower is better for Loss, PPL, and Peak Memory; higher is better for Tokens/sec.}
\label{tab:model_comparison}
\begin{tabular}{lcccc}
\toprule
Model & Loss ↓ & PPL ↓ & Tok/s ↑ & Peak MB ↓ \\
\midrule
dense & 1.102 ± 0.000 & 3.010 ± 0.000 & 420.300 ± 0.000 & 9100.000 ± 0.000 \\
sparse & 1.129 ± 0.000 & 3.100 ± 0.000 & 721.000 ± 0.000 & 7400.000 ± 0.000 \\
\bottomrule
\end{tabular}
\end{table}
```

## Troubleshooting

### Metric Names Don't Match

If the script can't find your metrics, it will show available keys. Update `KEY_MAP` in `make_metrics_csv.py`:

```python
KEY_MAP = {
    "loss": "your_actual_loss_key",
    "ppl": "your_actual_ppl_key", 
    "speed": "your_actual_speed_key",
    "mem": "your_actual_memory_key"
}
```

### Run IDs Don't Work

Update `RUN_IDS` in `make_metrics_csv.py` with your actual run IDs from the W&B UI URLs.

### Authentication Issues

```bash
wandb login --relogin
```

## Adding More Metrics

To extract additional metrics, edit `KEY_MAP` in `make_metrics_csv.py`:

```python
KEY_MAP = {
    "loss": "final_val_loss",
    "ppl": "final_val_ppl", 
    "speed": "tokens_per_sec",
    "mem": "peak_gpu_mem_MB",
    "params": "model_parameters",  # Add more metrics
    "flops": "total_flops"
}
```

Then update the CSV fieldnames and table headers accordingly.

## Alternative: Manual CSV Creation

If the W&B API doesn't work, you can manually create `metrics.csv`:

```csv
model,loss,ppl,speed,mem
dense,1.102,3.01,420.3,9100
sparse,1.129,3.10,721.0,7400
```

Then run just the table generation:
```bash
python generate_table.py
``` 