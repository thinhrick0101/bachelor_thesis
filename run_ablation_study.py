#!/usr/bin/env python3
"""
4-Way Ablation Study for Sparse Transformer
Addresses Major #3 reviewer feedback: systematic ablation of attention clusters

Usage: python run_ablation_study.py
"""

import subprocess
import sys
import time
import os
from pathlib import Path

# Configuration
ABLATION_CONFIGS = [
    {"subset": "0123", "name": "full_sparse", "description": "All clusters (baseline)"},
    {"subset": "0", "name": "cluster_0_only", "description": "Focused-local only"},
    {"subset": "1", "name": "cluster_1_only", "description": "Strided only"},
    {"subset": "2", "name": "cluster_2_only", "description": "Global-anchor only"},
    {"subset": "3", "name": "cluster_3_only", "description": "Wider-local only"},
]

BASE_ARGS = [
    "--num_epochs", "1",            # One epoch for quick comparison
    "--seed", "999",                # Consistent seed
    "--batch_size", "16",           # Smaller batch for speed
    "--learning_rate", "1e-4",
    "--wandb_project", "sparse-transformer-ablation",
]

def run_single_ablation(config):
    """Run a single ablation experiment"""
    subset = config["subset"]
    name = config["name"]
    description = config["description"]
    
    print(f"\n🚀 Running ablation: {name}")
    print(f"   Subset: {subset} - {description}")
    print("="*60)
    
    # Build command
    cmd = [
        sys.executable, "train_sparse_transformer.py",
        "--mask_subset", subset,
        "--wandb_run_name", f"abl_{subset}",
        *BASE_ARGS
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run the training
        result = subprocess.run(cmd, 
                              capture_output=True, 
                              text=True, 
                              timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            print(f"✅ SUCCESS: {name} completed")
            return True
        else:
            print(f"❌ FAILED: {name}")
            print(f"STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT: {name} took too long")
        return False
    except Exception as e:
        print(f"❌ ERROR: {name} - {e}")
        return False

def main():
    print("🔬 4-Way Ablation Study for Sparse Transformer")
    print("=" * 60)
    print("This will run 5 experiments:")
    for config in ABLATION_CONFIGS:
        print(f"  - {config['subset']}: {config['description']}")
    
    print(f"\nEach run will train for 1 epoch with seed=999")
    print(f"Results will be logged to W&B project: sparse-transformer-ablation")
    
    # Check if training script exists
    if not os.path.exists("train_sparse_transformer.py"):
        print("❌ Error: train_sparse_transformer.py not found")
        print("Make sure you're in the correct directory")
        return False
    
    print("\n🚀 Starting ablation study automatically...")
    
    # Run all ablations
    results = {}
    start_time = time.time()
    
    for config in ABLATION_CONFIGS:
        success = run_single_ablation(config)
        results[config["subset"]] = success
        
        if success:
            print(f"✅ Completed: {config['subset']}")
        else:
            print(f"❌ Failed: {config['subset']}")
        
        # Small delay between runs
        time.sleep(5)
    
    # Summary
    total_time = time.time() - start_time
    successful = sum(results.values())
    
    print(f"\n🏁 ABLATION STUDY COMPLETE!")
    print("=" * 60)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Successful runs: {successful}/{len(ABLATION_CONFIGS)}")
    
    for subset, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} Subset {subset}")
    
    if successful == len(ABLATION_CONFIGS):
        print(f"\n🎉 All runs completed successfully!")
        print(f"Next steps:")
        print(f"1. Run 'python collect_ablation_results.py' to gather metrics")
        print(f"2. Generate figures/tables for your thesis")
    else:
        print(f"\n⚠️  Some runs failed. Check the logs above.")
        print(f"You may need to re-run failed experiments manually.")
    
    return successful == len(ABLATION_CONFIGS)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 