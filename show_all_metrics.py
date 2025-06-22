#!/usr/bin/env python3
"""
Script to show all available metrics from W&B runs
Usage: python show_all_metrics.py
"""

import wandb
import json

# Project configuration
PROJECTS = {
    "dense": "dense-transformer-training",
    "sparse": "sparse-transformer-training"
}

RUN_IDS = {
    "dense": "w05vwtbq",
    "sparse": "vitzo6qf"
}

def show_metrics():
    print("📊 Available metrics in your W&B runs")
    print("="*60)
    
    api = wandb.Api()
    
    for model, run_id in RUN_IDS.items():
        try:
            project = PROJECTS[model]
            print(f"\n🔍 {model.upper()} run: {project}/{run_id}")
            
            run = api.run(f"{project}/{run_id}")
            
            # Get all summary metrics
            summary_keys = list(run.summary.keys())
            print(f"📈 Found {len(summary_keys)} summary metrics:")
            
            # Filter and categorize metrics
            loss_metrics = [k for k in summary_keys if 'loss' in k.lower()]
            ppl_metrics = [k for k in summary_keys if any(x in k.lower() for x in ['ppl', 'perplexity'])]
            speed_metrics = [k for k in summary_keys if any(x in k.lower() for x in ['speed', 'throughput', 'tokens', 'sec'])]
            memory_metrics = [k for k in summary_keys if any(x in k.lower() for x in ['mem', 'memory', 'gpu', 'ram'])]
            
            print(f"\n  🎯 LOSS metrics ({len(loss_metrics)}):")
            for metric in loss_metrics:
                value = run.summary.get(metric, "N/A")
                print(f"    {metric}: {value}")
            
            print(f"\n  🎯 PERPLEXITY metrics ({len(ppl_metrics)}):")
            for metric in ppl_metrics:
                value = run.summary.get(metric, "N/A")
                print(f"    {metric}: {value}")
            
            print(f"\n  🎯 SPEED metrics ({len(speed_metrics)}):")
            for metric in speed_metrics:
                value = run.summary.get(metric, "N/A")
                print(f"    {metric}: {value}")
            
            print(f"\n  🎯 MEMORY metrics ({len(memory_metrics)}):")
            for metric in memory_metrics:
                value = run.summary.get(metric, "N/A")
                print(f"    {metric}: {value}")
            
            # Show other interesting metrics
            other_metrics = [k for k in summary_keys if not any(x in k.lower() for x in 
                           ['loss', 'ppl', 'perplexity', 'speed', 'throughput', 'tokens', 'sec', 'mem', 'memory', 'gpu', 'ram', 'gradient', '_'])]
            
            if other_metrics:
                print(f"\n  📊 OTHER metrics (first 10 of {len(other_metrics)}):")
                for metric in other_metrics[:10]:
                    value = run.summary.get(metric, "N/A")
                    print(f"    {metric}: {value}")
                if len(other_metrics) > 10:
                    print(f"    ... and {len(other_metrics) - 10} more")
            
        except Exception as e:
            print(f"❌ Error fetching {model} run: {e}")
    
    print(f"\n{'='*60}")
    print("🔧 RECOMMENDED KEY_MAP based on available metrics:")
    print("Copy this into make_metrics_csv.py:")
    print()
    print("KEY_MAP = {")
    
    # Try to suggest the best metrics
    all_runs_data = {}
    for model, run_id in RUN_IDS.items():
        try:
            project = PROJECTS[model]
            run = api.run(f"{project}/{run_id}")
            all_runs_data[model] = run.summary
        except:
            continue
    
    # Find best loss metric
    loss_candidates = []
    for model, data in all_runs_data.items():
        loss_candidates.extend([k for k in data.keys() if 'loss' in k.lower() and not 'batch' in k.lower()])
    
    best_loss = None
    for candidate in ['val_loss', 'validation_loss', 'final_val_loss', 'test_loss']:
        if any(candidate in loss_candidates for loss_candidates in [loss_candidates]):
            best_loss = candidate
            break
    if not best_loss and loss_candidates:
        best_loss = loss_candidates[0]
    
    print(f'    "loss": "{best_loss or "batch_loss"}",')
    print(f'    "ppl": "batch_perplexity",  # or calculate from loss: exp(loss)')
    print(f'    "speed": "forward_pass_latency_ms",  # or create throughput metric')
    print(f'    "mem": "_runtime",  # no memory metric found - may need manual calculation')
    print("}")

def main():
    try:
        show_metrics()
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you're logged into W&B: wandb login")

if __name__ == "__main__":
    main() 