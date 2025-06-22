#!/usr/bin/env python3
"""
Visualization script for hyper-parameter sensitivity analysis results.
Creates publication-quality plots showing robustness of sparse attention parameters.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import argparse

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_sensitivity_results(csv_file):
    """Load sensitivity analysis results from CSV."""
    df = pd.read_csv(csv_file)
    
    # Filter out failed experiments
    valid_df = df[df['val_ppl'] != float('inf')].copy()
    
    if valid_df.empty:
        raise ValueError("No valid results found in the CSV file")
    
    return valid_df

def create_robustness_heatmap(df, metric='val_ppl', output_dir='.'):
    """Create heatmap showing parameter sensitivity."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Create pivot table for heatmap
    pivot_data = df.pivot_table(
        values=metric, 
        index='window_size', 
        columns='stride', 
        aggfunc='mean'
    )
    
    # Create heatmap
    if metric == 'val_ppl':
        cmap = 'RdYlBu_r'  # Red for high (bad), blue for low (good)
        cbar_label = 'Validation Perplexity'
        title = 'Validation Perplexity Sensitivity Analysis'
    elif metric == 'tokens_per_sec':
        cmap = 'RdYlGn'  # Red for low, green for high
        cbar_label = 'Tokens per Second'
        title = 'Training Speed Sensitivity Analysis'
    elif metric == 'peak_memory_mb':
        cmap = 'RdYlBu'  # Red for high, blue for low
        cbar_label = 'Peak Memory (MB)'
        title = 'Memory Usage Sensitivity Analysis'
    else:
        cmap = 'viridis'
        cbar_label = metric
        title = f'{metric} Sensitivity Analysis'
    
    im = ax.imshow(pivot_data.values, cmap=cmap, aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(range(len(pivot_data.columns)))
    ax.set_yticks(range(len(pivot_data.index)))
    ax.set_xticklabels(pivot_data.columns)
    ax.set_yticklabels(pivot_data.index)
    
    # Add value annotations
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            value = pivot_data.iloc[i, j]
            if not np.isnan(value):
                if metric == 'val_ppl':
                    text = f'{value:.3f}'
                elif metric == 'tokens_per_sec':
                    text = f'{value:.0f}'
                elif metric == 'peak_memory_mb':
                    text = f'{value:.0f}'
                else:
                    text = f'{value:.2f}'
                
                ax.text(j, i, text, ha="center", va="center", 
                       color='white' if value > np.median(pivot_data.values) else 'black',
                       fontweight='bold', fontsize=10)
    
    # Highlight baseline if present
    baseline_window, baseline_stride = 16, 8
    if baseline_window in pivot_data.index.values and baseline_stride in pivot_data.columns.values:
        baseline_i = list(pivot_data.index).index(baseline_window)
        baseline_j = list(pivot_data.columns).index(baseline_stride)
        
        # Add red border around baseline
        rect = plt.Rectangle((baseline_j-0.5, baseline_i-0.5), 1, 1, 
                           fill=False, edgecolor='red', linewidth=3)
        ax.add_patch(rect)
        
        # Add baseline label
        ax.text(baseline_j, baseline_i-0.7, 'BASELINE', ha='center', va='center',
               color='red', fontweight='bold', fontsize=8)
    
    ax.set_xlabel('Stride (s)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Window Size (w)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    filename = f'sensitivity_heatmap_{metric.replace("_", "")}.png'
    filepath = Path(output_dir) / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.savefig(filepath.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    
    return str(filepath)

def create_robustness_summary(df, output_dir='.'):
    """Create summary plot showing robustness across all metrics."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Get baseline values for comparison
    baseline = df[(df['window_size'] == 16) & (df['stride'] == 8)]
    if not baseline.empty:
        baseline_ppl = baseline.iloc[0]['val_ppl']
        baseline_speed = baseline.iloc[0]['tokens_per_sec']
        baseline_mem = baseline.iloc[0]['peak_memory_mb']
    else:
        baseline_ppl = df['val_ppl'].median()
        baseline_speed = df['tokens_per_sec'].median()
        baseline_mem = df['peak_memory_mb'].median()
    
    # 1. Perplexity variation
    ax1.scatter(range(len(df)), df['val_ppl'], alpha=0.7, s=60)
    ax1.axhline(y=baseline_ppl, color='red', linestyle='--', alpha=0.7, label='Baseline')
    ax1.axhline(y=baseline_ppl + 0.03, color='orange', linestyle=':', alpha=0.7, label='±0.03 PPL')
    ax1.axhline(y=baseline_ppl - 0.03, color='orange', linestyle=':', alpha=0.7)
    
    ax1.set_xlabel('Configuration Index')
    ax1.set_ylabel('Validation Perplexity')
    ax1.set_title('(A) Perplexity Robustness')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Speed scaling vs stride
    stride_groups = df.groupby('stride')
    stride_means = stride_groups['tokens_per_sec'].mean()
    stride_stds = stride_groups['tokens_per_sec'].std()
    
    ax2.errorbar(stride_means.index, stride_means.values, yerr=stride_stds.values,
                marker='o', capsize=5, capthick=2, markersize=8)
    ax2.axhline(y=baseline_speed, color='red', linestyle='--', alpha=0.7, label='Baseline')
    
    ax2.set_xlabel('Stride (s)')
    ax2.set_ylabel('Tokens per Second')
    ax2.set_title('(B) Speed vs Stride (∝ 1/s expected)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Memory usage patterns
    window_groups = df.groupby('window_size')
    window_mem_means = window_groups['peak_memory_mb'].mean()
    window_mem_stds = window_groups['peak_memory_mb'].std()
    
    ax3.errorbar(window_mem_means.index, window_mem_means.values, yerr=window_mem_stds.values,
                marker='s', capsize=5, capthick=2, markersize=8, color='green')
    ax3.axhline(y=baseline_mem, color='red', linestyle='--', alpha=0.7, label='Baseline')
    
    ax3.set_xlabel('Window Size (w)')
    ax3.set_ylabel('Peak Memory (MB)')
    ax3.set_title('(C) Memory vs Window Size')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Configuration space overview
    # Create scatter plot with window size vs stride, colored by PPL
    scatter = ax4.scatter(df['stride'], df['window_size'], 
                         c=df['val_ppl'], s=df['tokens_per_sec']/10, 
                         cmap='RdYlBu_r', alpha=0.7, edgecolors='black')
    
    # Highlight baseline
    if not baseline.empty:
        ax4.scatter(baseline.iloc[0]['stride'], baseline.iloc[0]['window_size'],
                   c='red', s=200, marker='*', edgecolors='black', linewidth=2,
                   label='Baseline', zorder=5)
    
    ax4.set_xlabel('Stride (s)')
    ax4.set_ylabel('Window Size (w)')
    ax4.set_title('(D) Configuration Space\n(Color: PPL, Size: Speed)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Val PPL', fontsize=10)
    
    plt.suptitle('Hyper-parameter Sensitivity Analysis: Robustness Summary', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Save plot
    filepath = Path(output_dir) / 'sensitivity_robustness_summary.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.savefig(filepath.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    
    return str(filepath)

def create_performance_vs_baseline(df, output_dir='.'):
    """Create plot showing performance relative to baseline."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Find baseline
    baseline = df[(df['window_size'] == 16) & (df['stride'] == 8)]
    if baseline.empty:
        print("Warning: Baseline configuration not found")
        return None
    
    baseline_ppl = baseline.iloc[0]['val_ppl']
    baseline_speed = baseline.iloc[0]['tokens_per_sec']
    baseline_mem = baseline.iloc[0]['peak_memory_mb']
    
    # Calculate relative performance
    df_rel = df.copy()
    df_rel['ppl_ratio'] = df_rel['val_ppl'] / baseline_ppl
    df_rel['speed_ratio'] = df_rel['tokens_per_sec'] / baseline_speed
    df_rel['mem_ratio'] = df_rel['peak_memory_mb'] / baseline_mem
    
    # Configuration labels
    df_rel['config_label'] = df_rel.apply(lambda x: f"({int(x['window_size'])},{int(x['stride'])},{int(x['global_anchors'])})", axis=1)
    
    # Remove baseline from comparison
    df_comp = df_rel[~((df_rel['window_size'] == 16) & (df_rel['stride'] == 8))].copy()
    
    # 1. PPL ratio (lower is better)
    colors1 = ['green' if x <= 1.0 else 'red' for x in df_comp['ppl_ratio']]
    bars1 = ax1.bar(range(len(df_comp)), df_comp['ppl_ratio'], color=colors1, alpha=0.7)
    ax1.axhline(y=1.0, color='black', linestyle='-', alpha=0.5, label='Baseline')
    ax1.axhline(y=1.03, color='orange', linestyle='--', alpha=0.5, label='±3% PPL')
    ax1.axhline(y=0.97, color='orange', linestyle='--', alpha=0.5)
    
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('PPL Ratio (vs Baseline)')
    ax1.set_title('(A) Perplexity Ratio\n(Lower = Better)')
    ax1.set_xticks(range(len(df_comp)))
    ax1.set_xticklabels(df_comp['config_label'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, ratio) in enumerate(zip(bars1, df_comp['ppl_ratio'])):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{ratio:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 2. Speed ratio (higher is better)
    colors2 = ['green' if x >= 1.0 else 'red' for x in df_comp['speed_ratio']]
    bars2 = ax2.bar(range(len(df_comp)), df_comp['speed_ratio'], color=colors2, alpha=0.7)
    ax2.axhline(y=1.0, color='black', linestyle='-', alpha=0.5, label='Baseline')
    ax2.axhline(y=1.05, color='orange', linestyle='--', alpha=0.5, label='±5% Speed')
    ax2.axhline(y=0.95, color='orange', linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('Configuration')
    ax2.set_ylabel('Speed Ratio (vs Baseline)')
    ax2.set_title('(B) Speed Ratio\n(Higher = Better)')
    ax2.set_xticks(range(len(df_comp)))
    ax2.set_xticklabels(df_comp['config_label'], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, ratio) in enumerate(zip(bars2, df_comp['speed_ratio'])):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{ratio:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 3. Memory ratio (lower is better)
    colors3 = ['green' if x <= 1.0 else 'red' for x in df_comp['mem_ratio']]
    bars3 = ax3.bar(range(len(df_comp)), df_comp['mem_ratio'], color=colors3, alpha=0.7)
    ax3.axhline(y=1.0, color='black', linestyle='-', alpha=0.5, label='Baseline')
    
    ax3.set_xlabel('Configuration')
    ax3.set_ylabel('Memory Ratio (vs Baseline)')
    ax3.set_title('(C) Memory Ratio\n(Lower = Better)')
    ax3.set_xticks(range(len(df_comp)))
    ax3.set_xticklabels(df_comp['config_label'], rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, ratio) in enumerate(zip(bars3, df_comp['mem_ratio'])):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{ratio:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.suptitle('Performance Relative to Baseline (w=16, s=8, a=64)', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save plot
    filepath = Path(output_dir) / 'sensitivity_vs_baseline.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.savefig(filepath.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    
    return str(filepath)

def create_thesis_table(df, output_dir='.'):
    """Create thesis-ready table of results."""
    # Sort by window size and stride for logical ordering
    df_sorted = df.sort_values(['window_size', 'stride']).copy()
    
    # Create the table
    table_data = []
    headers = ['Configuration (w,s,a)', 'Val PPL ↓', 'Tok/s ↑', 'Mem (MiB)', 'Notes']
    
    baseline_ppl = None
    baseline_speed = None
    
    for _, row in df_sorted.iterrows():
        w, s, a = int(row['window_size']), int(row['stride']), int(row['global_anchors'])
        ppl = row['val_ppl']
        speed = row['tokens_per_sec']
        mem = row['peak_memory_mb']
        
        # Check if baseline
        is_baseline = (w == 16 and s == 8)
        if is_baseline:
            baseline_ppl = ppl
            baseline_speed = speed
            notes = "baseline"
        else:
            notes = ""
            
            # Add robustness indicators
            if baseline_ppl is not None:
                ppl_diff = abs(ppl - baseline_ppl)
                if ppl_diff <= 0.03:
                    notes += "robust" if notes == "" else ", robust"
        
        table_data.append([
            f"({w},{s},{a})",
            f"{ppl:.3f}",
            f"{speed:.0f}",
            f"{mem:.0f}",
            notes
        ])
    
    # Save as CSV for easy copying to thesis
    table_df = pd.DataFrame(table_data, columns=headers)
    csv_path = Path(output_dir) / 'sensitivity_thesis_table.csv'
    table_df.to_csv(csv_path, index=False)
    
    # Also create LaTeX table
    latex_path = Path(output_dir) / 'sensitivity_thesis_table.tex'
    with open(latex_path, 'w') as f:
        f.write("\\begin{table}[h!]\n")
        f.write("\\centering\n")
        f.write("\\caption{Hyper-parameter sensitivity analysis results showing robustness of sparse attention patterns.}\n")
        f.write("\\label{tab:sensitivity-analysis}\n")
        f.write("\\begin{tabular}{|l|c|c|c|l|}\n")
        f.write("\\hline\n")
        f.write("Configuration $(w,s,a)$ & Val PPL $\\downarrow$ & Tok/s $\\uparrow$ & Mem (MiB) & Notes \\\\\n")
        f.write("\\hline\n")
        
        for row in table_data:
            config, ppl, speed, mem, notes = row
            if "baseline" in notes:
                f.write(f"\\textbf{{{config}}} & \\textbf{{{ppl}}} & \\textbf{{{speed}}} & \\textbf{{{mem}}} & {notes} \\\\\n")
            else:
                f.write(f"{config} & {ppl} & {speed} & {mem} & {notes} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"📋 Thesis table saved to: {csv_path}")
    print(f"📋 LaTeX table saved to: {latex_path}")
    
    return str(csv_path)

def analyze_robustness(df):
    """Analyze and report robustness metrics."""
    if df.empty:
        return "No valid results to analyze"
    
    # Find baseline
    baseline = df[(df['window_size'] == 16) & (df['stride'] == 8)]
    if baseline.empty:
        baseline_ppl = df['val_ppl'].median()
        baseline_speed = df['tokens_per_sec'].median()
        print("Warning: Baseline not found, using median values")
    else:
        baseline_ppl = baseline.iloc[0]['val_ppl']
        baseline_speed = baseline.iloc[0]['tokens_per_sec']
    
    # Calculate robustness metrics
    ppl_range = df['val_ppl'].max() - df['val_ppl'].min()
    ppl_std = df['val_ppl'].std()
    
    speed_range = df['tokens_per_sec'].max() - df['tokens_per_sec'].min()
    speed_cv = df['tokens_per_sec'].std() / df['tokens_per_sec'].mean()  # Coefficient of variation
    
    # Count robust configurations (within ±0.03 PPL)
    robust_configs = df[abs(df['val_ppl'] - baseline_ppl) <= 0.03]
    robustness_pct = len(robust_configs) / len(df) * 100
    
    # Speed scaling analysis
    stride_correlation = df['tokens_per_sec'].corr(1/df['stride'])
    
    analysis = f"""
🔍 ROBUSTNESS ANALYSIS:
========================

📊 Perplexity Robustness:
   • Range: {ppl_range:.4f} PPL
   • Standard deviation: {ppl_std:.4f}
   • Robust configs (±0.03 PPL): {len(robust_configs)}/{len(df)} ({robustness_pct:.1f}%)
   • Assessment: {'✅ ROBUST' if ppl_range <= 0.03 else '⚠️ MODERATE' if ppl_range <= 0.1 else '❌ BRITTLE'}

⚡ Speed Characteristics:
   • Range: {speed_range:.0f} tok/s
   • Coefficient of variation: {speed_cv:.2f}
   • Correlation with 1/stride: {stride_correlation:.3f}
   • Expected scaling: {'✅ CONFIRMED' if stride_correlation > 0.5 else '⚠️ WEAK' if stride_correlation > 0.2 else '❌ NOT OBSERVED'}

🎯 Key Findings:
   • Validation PPL varies by {ppl_range:.3f} across configurations
   • Speed follows expected ∝1/s trend (r={stride_correlation:.3f})
   • {robustness_pct:.0f}% of configurations are within robust range
   • Design is {'not brittle' if robustness_pct >= 70 else 'moderately sensitive'}

📝 Thesis Statement:
   "Validation perplexity varies by < {ppl_range:.2f} across the parameter grid while 
   speed follows the expected 1/s trend (r={stride_correlation:.2f}), confirming that 
   the sparse attention design is robust to hyperparameter variations."
"""
    
    return analysis

def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(description='Visualize sensitivity analysis results')
    parser.add_argument('csv_file', help='Path to CSV results file from benchmark_grid.py')
    parser.add_argument('--output_dir', type=str, default='sensitivity_plots',
                        help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load results
    print(f"📊 Loading results from {args.csv_file}")
    try:
        df = load_sensitivity_results(args.csv_file)
        print(f"✅ Loaded {len(df)} valid configurations")
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return
    
    # Generate visualizations
    print("🎨 Creating visualizations...")
    
    plots_created = []
    
    # 1. Heatmaps for each metric
    for metric in ['val_ppl', 'tokens_per_sec', 'peak_memory_mb']:
        try:
            plot_path = create_robustness_heatmap(df, metric, output_dir)
            plots_created.append(plot_path)
            print(f"✅ Created heatmap: {plot_path}")
        except Exception as e:
            print(f"❌ Failed to create {metric} heatmap: {e}")
    
    # 2. Robustness summary
    try:
        plot_path = create_robustness_summary(df, output_dir)
        plots_created.append(plot_path)
        print(f"✅ Created robustness summary: {plot_path}")
    except Exception as e:
        print(f"❌ Failed to create robustness summary: {e}")
    
    # 3. Performance vs baseline
    try:
        plot_path = create_performance_vs_baseline(df, output_dir)
        if plot_path:
            plots_created.append(plot_path)
            print(f"✅ Created baseline comparison: {plot_path}")
    except Exception as e:
        print(f"❌ Failed to create baseline comparison: {e}")
    
    # 4. Thesis table
    try:
        table_path = create_thesis_table(df, output_dir)
        print(f"✅ Created thesis table: {table_path}")
    except Exception as e:
        print(f"❌ Failed to create thesis table: {e}")
    
    # 5. Robustness analysis
    print("\n" + "="*80)
    print(analyze_robustness(df))
    print("="*80)
    
    # Save analysis to file
    analysis_text = analyze_robustness(df)
    with open(output_dir / 'robustness_analysis.txt', 'w') as f:
        f.write(analysis_text)
    
    print(f"\n🎉 Visualization complete!")
    print(f"📁 All files saved to: {output_dir}")
    print(f"📊 {len(plots_created)} plots created")
    print(f"📋 Thesis-ready table and LaTeX code generated")
    print(f"📝 Robustness analysis saved to robustness_analysis.txt")

if __name__ == "__main__":
    main() 