#!/usr/bin/env python3
"""
Generate Table 2.1 for MC 3.1 deliverables
Creates a Word-friendly TSV file with mean ± std statistics per layer
"""

import pandas as pd
import sys
import os

def generate_table_2_1(input_file="head_metrics.csv", output_file="table_2_1.tsv"):
    """
    Generate Table 2.1 with mean ± standard deviation per layer
    
    Args:
        input_file: CSV file with columns: layer, entropy, sparsity, distance
        output_file: Output TSV file for Word import
    """
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file '{input_file}' not found!")
        print(f"Expected columns: layer, entropy, sparsity, distance")
        return False
    
    try:
        # Read the data
        print(f"📊 Reading data from {input_file}...")
        df = pd.read_csv(input_file)
        
        # Verify required columns
        required_cols = ['layer', 'entropy', 'sparsity', 'distance']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ Error: Missing columns: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")
            return False
        
        # Generate summary statistics
        print("🔢 Computing mean ± std per layer...")
        summary = df.groupby("layer").agg(["mean", "std"]).round(3)
        
        # Save as TSV for Word import
        summary.to_csv(output_file, sep="\t")
        print(f"✅ Table saved as {output_file}")
        
        # Display preview
        print("\n📋 Preview of Table 2.1:")
        print("=" * 60)
        print(summary)
        print("=" * 60)
        
        # Instructions for Word
        print(f"\n📝 To import into Word:")
        print(f"1. Open {output_file} in text editor → Ctrl+A → Ctrl+C")
        print(f"2. In Word → Paste → 'Keep Text Only'")
        print(f"3. Select pasted text → Insert ▸ Table ▸ Convert Text to Table")
        print(f"4. Choose 'Tabs' as delimiter → OK")
        print(f"5. Bold header row and center numeric columns")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing data: {e}")
        return False

if __name__ == "__main__":
    # Allow custom input file via command line
    input_file = sys.argv[1] if len(sys.argv) > 1 else "head_metrics.csv"
    
    print("🎯 MC 3.1 Table Generator")
    print("=" * 40)
    
    success = generate_table_2_1(input_file)
    
    if success:
        print("\n✅ Table 2.1 generation complete!")
        print("Ready for Word import following the instructions above.")
    else:
        print("\n❌ Table generation failed. Please check your data file.")
        sys.exit(1) 