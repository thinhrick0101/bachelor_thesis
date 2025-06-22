#!/usr/bin/env python3
"""
Script to find available W&B runs and their IDs
Usage: python find_wandb_runs.py
"""

import wandb
import sys
from datetime import datetime

# Your W&B username from the error message
USERNAME = "nguyen-djuc-thinh-vu-amsterdam"

# Project names
PROJECTS = [
    "dense-transformer-training",
    "sparse-transformer-training"
]

def find_runs():
    print(f"Searching for W&B runs under username: {USERNAME}")
    print("="*60)
    
    api = wandb.Api()
    
    for project_name in PROJECTS:
        try:
            project_path = f"{USERNAME}/{project_name}"
            print(f"\n🔍 Searching project: {project_path}")
            
            # Get runs from this project
            runs = api.runs(project_path)
            
            if not runs:
                print(f"  ❌ No runs found in {project_name}")
                continue
                
            print(f"  ✅ Found {len(list(runs))} runs in {project_name}")
            
            # Reset the iterator and get details
            runs = api.runs(project_path)
            
            for i, run in enumerate(runs):
                if i >= 5:  # Limit to first 5 runs
                    print(f"  ... (showing first 5 of {len(list(api.runs(project_path)))} total runs)")
                    break
                    
                # Get run info
                state = run.state
                created_at = run.created_at
                if created_at:
                    if isinstance(created_at, str):
                        created_str = created_at
                    else:
                        created_str = created_at.strftime("%Y-%m-%d %H:%M")
                else:
                    created_str = "Unknown"
                
                print(f"\n  Run {i+1}:")
                print(f"    ID: {run.id}")
                print(f"    Name: {run.name}")
                print(f"    State: {state}")
                print(f"    Created: {created_str}")
                print(f"    URL: {run.url}")
                
                # Show some summary metrics if available
                if run.summary:
                    print(f"    Available metrics: {list(run.summary.keys())[:10]}")
                    if len(run.summary.keys()) > 10:
                        print(f"    ... and {len(run.summary.keys()) - 10} more")
                else:
                    print(f"    No summary metrics available")
                    
        except Exception as e:
            print(f"  ❌ Error accessing {project_name}: {e}")
            
    print(f"\n{'='*60}")
    print("Instructions:")
    print("1. Copy the Run ID from your desired runs above")
    print("2. Update RUN_IDS in make_metrics_csv.py with the correct IDs")
    print("3. Check the 'Available metrics' to update KEY_MAP if needed")

def main():
    try:
        find_runs()
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're logged into W&B: wandb login")
        print("2. Check your internet connection")
        print("3. Verify your project names are correct")

if __name__ == "__main__":
    main() 