#!/usr/bin/env python3
"""
Script to pull metrics directly from W&B runs and create metrics.csv
Usage: python make_metrics_csv.py
"""

import wandb
import csv
import sys

# Project names from your codebase
PROJECTS = {
    "dense": "dense-transformer-training",
    "sparse": "sparse-transformer-training"
}

# Run IDs from W&B discovery
RUN_IDS = {
    "dense": "w05vwtbq",
    "sparse": "vitzo6qf"
}

# Keys you want from run.summary - based on actual available metrics
KEY_MAP = {
    "loss": "val_loss",              # validation loss
    "ppl": "val_perplexity",         # validation perplexity
    "speed": "throughput_samples_per_sec",  # samples per second throughput
    "mem": "_runtime"                # no memory metric available - using runtime as placeholder
}

def main():
    print("Pulling metrics from W&B runs...")
    
    rows = []
    api = wandb.Api()
    
    for model, run_id in RUN_IDS.items():
        try:
            project = PROJECTS[model]
            print(f"Fetching {model} run: {project}/{run_id}")
            
            run = api.run(f"{project}/{run_id}")
            print(f"Run summary keys: {list(run.summary.keys())}")
            
            row = {"model": model}
            
            for short_key, wb_key in KEY_MAP.items():
                if wb_key in run.summary:
                    value = run.summary[wb_key]
                    row[short_key] = round(value, 3) if isinstance(value, (int, float)) else value
                else:
                    print(f"Warning: Key '{wb_key}' not found in {model} run summary")
                    # Try some common alternatives
                    alternatives = {
                        "final_val_loss": ["val_loss", "validation_loss", "loss"],
                        "final_val_ppl": ["val_ppl", "validation_ppl", "ppl", "perplexity"],
                        "tokens_per_sec": ["throughput_samples_per_sec", "throughput", "speed"],
                        "peak_gpu_mem_MB": ["gpu_mem_MB", "memory_MB", "peak_memory"]
                    }
                    
                    found = False
                    for alt in alternatives.get(wb_key, []):
                        if alt in run.summary:
                            value = run.summary[alt]
                            row[short_key] = round(value, 3) if isinstance(value, (int, float)) else value
                            print(f"  Used alternative key '{alt}' for {short_key}")
                            found = True
                            break
                    
                    if not found:
                        row[short_key] = "N/A"
                        print(f"  No suitable key found for {short_key}")
            
            rows.append(row)
            print(f"✓ Successfully fetched {model} metrics")
            
        except Exception as e:
            print(f"Error fetching {model} run: {e}")
            # Add a placeholder row so we can still generate some output
            rows.append({
                "model": model,
                "loss": "ERROR",
                "ppl": "ERROR", 
                "speed": "ERROR",
                "mem": "ERROR"
            })
    
    # Write CSV
    csv_file = "metrics.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "loss", "ppl", "speed", "mem"])
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\n✓ {csv_file} created with content:")
    for row in rows:
        print(f"  {row}")
    
    return csv_file

if __name__ == "__main__":
    csv_file = main()
    print(f"\nNext steps:")
    print(f"1. Check {csv_file} for your metrics")
    print(f"2. Run 'python generate_table.py' to create LaTeX table") 