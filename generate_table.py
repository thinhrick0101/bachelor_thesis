#!/usr/bin/env python3
"""
Script to generate LaTeX and Markdown tables from metrics.csv
Usage: python generate_table.py
"""

import pandas as pd
import os

def format_value_with_error(value, error=0):
    """Format a value with ± error notation"""
    if isinstance(value, str) and value in ["N/A", "ERROR"]:
        return value
    try:
        return f"{float(value):.3f} ± {error:.3f}"
    except (ValueError, TypeError):
        return str(value)

def main():
    csv_file = "metrics.csv"
    
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Run 'python make_metrics_csv.py' first.")
        return
    
    print(f"Reading {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # Add "± 0" columns since only one run per model
    table_df = df.copy()
    for col in ["loss", "ppl", "speed", "mem"]:
        if col in table_df.columns:
            table_df[col] = table_df[col].apply(lambda x: format_value_with_error(x, 0))
    
    print("\nFormatted table:")
    print(table_df)
    
    # Generate LaTeX table
    latex_file = "table_metrics.tex"
    latex_content = table_df.to_latex(
        index=False,
        escape=False,  # Don't escape LaTeX special characters
        column_format="lcccc",
        header=["Model", "Loss ↓", "PPL ↓", "Tok/s ↑", "Peak MB ↓"]
    )
    
    # Clean up LaTeX output and add professional styling
    latex_content = latex_content.replace("\\toprule", "\\toprule")
    latex_content = latex_content.replace("\\midrule", "\\midrule") 
    latex_content = latex_content.replace("\\bottomrule", "\\bottomrule")
    
    # Add caption and label
    latex_with_caption = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Performance comparison between dense and sparse transformer models. Lower is better for Loss, PPL, and Peak Memory; higher is better for Tokens/sec.}}
\\label{{tab:model_comparison}}
{latex_content}
\\end{{table}}"""
    
    with open(latex_file, "w") as f:
        f.write(latex_with_caption)
    
    print(f"\n✓ LaTeX table saved to {latex_file}")
    
    # Generate Markdown table
    markdown_file = "table_metrics.md"
    markdown_content = table_df.to_markdown(index=False)
    
    # Add caption for Markdown
    markdown_with_caption = f"""# Model Performance Comparison

{markdown_content}

**Table**: Performance comparison between dense and sparse transformer models. Lower is better for Loss, PPL, and Peak Memory; higher is better for Tokens/sec.
"""
    
    with open(markdown_file, "w") as f:
        f.write(markdown_with_caption)
    
    print(f"✓ Markdown table saved to {markdown_file}")
    
    # Print raw CSV for quick reference
    print(f"\nRaw CSV content:")
    print(df.to_string(index=False))
    
    # Print LaTeX code for copy-paste
    print(f"\n" + "="*50)
    print("LaTeX code for thesis (copy-paste ready):")
    print("="*50)
    print(latex_with_caption)
    
    return latex_file, markdown_file

if __name__ == "__main__":
    main() 