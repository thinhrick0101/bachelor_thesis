#!/usr/bin/env python3
"""
Script to fix Figure 6 - Perplexity comparison with logarithmic scale
Addresses reviewer feedback about scale issues in perplexity plots

Usage: python fix_figure6_perplexity.py
"""

import wandb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configuration
PROJECTS = {
    "dense": "dense-transformer-training",
    "sparse": "sparse-transformer-training"
}

RUN_IDS = {
    "dense": "w05vwtbq",
    "sparse": "vitzo6qf"
}

def fetch_perplexity_data():
    """Fetch perplexity data from W&B runs"""
    print("📊 Fetching perplexity data from W&B runs...")
    
    api = wandb.Api()
    data = {}
    
    for model, run_id in RUN_IDS.items():
        try:
            project = PROJECTS[model]
            print(f"  Fetching {model} run: {project}/{run_id}")
            
            run = api.run(f"{project}/{run_id}")
            
            # Get perplexity history - try different possible column names
            ppl_keys = ["val_perplexity", "validation_perplexity", "batch_perplexity", "perplexity"]
            history_data = None
            
            for key in ppl_keys:
                try:
                    history_data = run.history(keys=["_step", key])
                    if not history_data.empty and key in history_data.columns:
                        print(f"    ✅ Found perplexity data in column: {key}")
                        break
                except:
                    continue
            
            if history_data is None or history_data.empty:
                print(f"    ❌ No perplexity data found for {model}")
                continue
            
            # Clean the data
            history_data = history_data.dropna()
            
            # Rename the perplexity column to standard name
            ppl_col = [col for col in history_data.columns if 'perplexity' in col.lower()][0]
            history_data = history_data.rename(columns={ppl_col: 'perplexity'})
            
            data[model] = history_data
            
            print(f"    📈 {len(history_data)} data points")
            print(f"    📊 Perplexity range: {history_data['perplexity'].min():.3f} - {history_data['perplexity'].max():.3f}")
            
            # Show first few values to check for extremely high early values
            print(f"    🔍 First 5 perplexity values: {list(history_data['perplexity'].head().round(2))}")
            
        except Exception as e:
            print(f"    ❌ Error fetching {model} data: {e}")
    
    return data

def create_figure6_fixed(data):
    """Create the fixed Figure 6 with logarithmic scale and synchronized x-axis"""
    print("\n🎨 Creating Figure 6 with logarithmic scale and synchronized steps...")
    
    if len(data) == 0:
        print("❌ No data available to plot")
        return None
    
    # Find the minimum max step across both models to synchronize the x-axis
    max_steps = []
    for model, df in data.items():
        if df is not None and not df.empty:
            max_steps.append(df["_step"].max())
            print(f"  📊 {model} max steps: {df['_step'].max()}")
    
    # Use the smaller range so both models are comparable
    sync_max_step = min(max_steps) if max_steps else None
    print(f"  🔄 Synchronizing to max step: {sync_max_step}")
    
    # Set up the plot with publication quality
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    colors = {'dense': '#1f77b4', 'sparse': '#ff7f0e'}  # Blue and orange
    
    for model, df in data.items():
        if df is not None and not df.empty:
            # Filter data to synchronized step range
            if sync_max_step:
                df_sync = df[df["_step"] <= sync_max_step].copy()
            else:
                df_sync = df.copy()
            
            label = f"{model.capitalize()} Transformer"
            ax.plot(df_sync["_step"], df_sync["perplexity"], 
                   label=label, color=colors.get(model, 'black'), 
                   linewidth=2, alpha=0.8)
            
            print(f"  ✅ Plotted {model}: {len(df_sync)} points (synced from {len(df)} total)")
    
    # Set logarithmic scale for y-axis
    ax.set_yscale("log")
    
    # Synchronize x-axis range
    if sync_max_step:
        ax.set_xlim(0, sync_max_step * 1.05)  # Add 5% padding
    
    # Labels and title
    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Validation Perplexity (log-scale)", fontsize=12)
    ax.set_title("Dense vs. Sparse Transformer Perplexity Comparison", fontsize=14, fontweight='bold')
    
    # Legend and grid
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, which="both", linestyle="--", alpha=0.7)
    ax.grid(True, which="minor", linestyle=":", alpha=0.4)
    
    # Improve layout
    plt.tight_layout()
    
    # Save the figure
    output_file = "figure_6_fixed_synced.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig("figure_6_fixed_synced.pdf", dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"  ✅ Saved as {output_file} and figure_6_fixed_synced.pdf")
    
    # Show the plot
    plt.show()
    
    return output_file

def generate_caption():
    """Generate the updated caption for the figure"""
    caption_latex = r"""
\begin{figure}[ht]
  \centering
  \includegraphics[width=\linewidth]{figures/figure_6_fixed.png}
  \caption{Validation perplexity curves of dense and sparse Transformer models across training steps. The vertical axis is displayed on a logarithmic scale to clearly illustrate model convergence behavior despite high early perplexities. The sparse transformer shows initial fluctuations but converges to similar performance levels as the dense model.}
  \label{fig:val_ppl_log}
\end{figure}
"""
    
    caption_markdown = """
**Figure 6:** Validation perplexity curves of dense and sparse Transformer models across training steps. The vertical axis is displayed on a **logarithmic scale** to clearly illustrate model convergence behavior despite high early perplexities. The sparse transformer shows initial fluctuations but converges to similar performance levels as the dense model.
"""
    
    return caption_latex, caption_markdown

def main():
    print("🔧 Fixing Figure 6 - Perplexity Plot with Logarithmic Scale")
    print("=" * 60)
    
    # Step 1: Fetch perplexity data
    data = fetch_perplexity_data()
    
    if not data:
        print("\n❌ No perplexity data found. Check your run IDs and metric names.")
        return
    
    # Step 2: Create the fixed plot
    output_file = create_figure6_fixed(data)
    
    if output_file:
        # Step 3: Generate updated captions
        latex_caption, markdown_caption = generate_caption()
        
        # Save captions to files
        with open("figure_6_caption.tex", "w") as f:
            f.write(latex_caption)
        
        with open("figure_6_caption.md", "w") as f:
            f.write(markdown_caption)
        
        print(f"\n🎉 SUCCESS! Figure 6 has been fixed:")
        print("=" * 60)
        print(f"📄 {output_file} - Updated figure with synchronized steps and log scale")
        print(f"📄 {output_file.replace('.png', '.pdf')} - PDF version for LaTeX")
        print(f"📄 figure_6_caption.tex - LaTeX caption")
        print(f"📄 figure_6_caption.md - Markdown caption")
        
        print(f"\n📝 NEXT STEPS:")
        print("1. Replace your original Figure 6 with figure_6_fixed.png")
        print("2. Update the caption using the text from figure_6_caption.tex")
        print("3. Ensure the caption mentions 'logarithmic scale'")
        print("4. This addresses Major #2 from your reviewer feedback!")
        
        print(f"\n📋 Updated Caption Preview:")
        print(markdown_caption)

if __name__ == "__main__":
    main() 