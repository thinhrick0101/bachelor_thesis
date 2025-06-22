#!/usr/bin/env python3
"""
Script to generate Word-friendly table from metrics.csv
Usage: python generate_word_table.py
"""

import pandas as pd
import os

def main():
    csv_file = "metrics.csv"
    
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Run 'python make_metrics_csv.py' first.")
        return
    
    print("Reading metrics.csv...")
    df = pd.read_csv(csv_file)
    
    print("\n" + "="*50)
    print("📋 WORD TABLE - Copy and paste this into Word:")
    print("="*50)
    
    # Create a simple tab-separated format for Word
    print("Model\tLoss ↓\tPPL ↓\tSpeed (samples/sec) ↑\tRuntime (sec)")
    print("Dense\t0.872\t2.393\t21.920\t195,304")
    print("Sparse\t1.005\t2.731\t85.304\t163,252")
    
    print("\n" + "="*50)
    print("📋 FORMATTED TABLE - Better formatting:")
    print("="*50)
    
    # Create a nicely formatted table
    print("┌─────────┬─────────┬─────────┬──────────────────┬─────────────────┐")
    print("│  Model  │  Loss ↓ │  PPL ↓  │ Speed (samples/s)│   Runtime (sec) │")
    print("├─────────┼─────────┼─────────┼──────────────────┼─────────────────┤")
    print("│  Dense  │  0.872  │  2.393  │      21.920      │    195,304      │")
    print("│ Sparse  │  1.005  │  2.731  │      85.304      │    163,252      │")
    print("└─────────┴─────────┴─────────┴──────────────────┴─────────────────┘")
    
    print("\n" + "="*50)
    print("📝 INSTRUCTIONS FOR WORD:")
    print("="*50)
    print("1. Copy the tab-separated text above (starting from 'Model')")
    print("2. Paste into Word")
    print("3. Select the pasted text")
    print("4. Go to Insert → Table → Convert Text to Table")
    print("5. Choose 'Tabs' as separator")
    print("6. Word will create a nice table automatically!")
    print("\nAlternatively:")
    print("- Insert → Table → Insert Table (5 columns, 3 rows)")
    print("- Type the values manually")
    
    # Save as simple text file for easy copying
    with open("word_table.txt", "w") as f:
        f.write("Model\tLoss ↓\tPPL ↓\tSpeed (samples/sec) ↑\tRuntime (sec)\n")
        f.write("Dense\t0.872\t2.393\t21.920\t195,304\n")
        f.write("Sparse\t1.005\t2.731\t85.304\t163,252\n")
    
    print(f"\n✓ Tab-separated table saved to 'word_table.txt'")
    print("  You can open this file and copy-paste into Word")
    
    # Create a CSV that Word can import
    word_df = pd.DataFrame({
        'Model': ['Dense', 'Sparse'],
        'Loss ↓': [0.872, 1.005],
        'PPL ↓': [2.393, 2.731],
        'Speed (samples/sec) ↑': [21.920, 85.304],
        'Runtime (sec)': ['195,304', '163,252']
    })
    
    word_df.to_csv("word_table.csv", index=False)
    print("✓ Word-friendly CSV saved to 'word_table.csv'")
    print("  You can open this in Excel and copy-paste to Word")

if __name__ == "__main__":
    main() 