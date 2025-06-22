#!/usr/bin/env python3
"""
Convenience script to create metrics.csv and generate tables in one go
Usage: python create_metrics_table.py
"""

import subprocess
import sys
import os

def run_script(script_name):
    """Run a Python script and return success status"""
    try:
        print(f"\n{'='*50}")
        print(f"Running {script_name}...")
        print('='*50)
        
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=False, 
                              text=True, 
                              cwd=os.path.dirname(os.path.abspath(__file__)))
        
        if result.returncode == 0:
            print(f"✓ {script_name} completed successfully")
            return True
        else:
            print(f"✗ {script_name} failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"✗ Error running {script_name}: {e}")
        return False

def main():
    print("Creating metrics table from W&B runs...")
    print("This will:")
    print("1. Pull metrics from your W&B runs")
    print("2. Create metrics.csv")
    print("3. Generate LaTeX and Markdown tables")
    
    # Check if wandb is available
    try:
        import wandb
        print("✓ wandb package found")
    except ImportError:
        print("✗ wandb package not found. Please install with: pip install wandb")
        return False
    
    # Check if pandas is available
    try:
        import pandas
        print("✓ pandas package found")
    except ImportError:
        print("✗ pandas package not found. Please install with: pip install pandas")
        return False
    
    # Step 1: Extract metrics from W&B
    success1 = run_script("make_metrics_csv.py")
    if not success1:
        print("\n✗ Failed to extract metrics from W&B")
        return False
    
    # Step 2: Generate tables
    success2 = run_script("generate_table.py")
    if not success2:
        print("\n✗ Failed to generate tables")
        return False
    
    print(f"\n{'='*50}")
    print("🎉 SUCCESS! All files created:")
    print("="*50)
    print("📄 metrics.csv - Raw metrics data")
    print("📄 table_metrics.tex - LaTeX table for thesis")
    print("📄 table_metrics.md - Markdown table for reference")
    print("\nYou can now:")
    print("1. Include table_metrics.tex in your thesis LaTeX document")
    print("2. View table_metrics.md for a preview")
    print("3. Edit metrics.csv if you need to make manual adjustments")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 