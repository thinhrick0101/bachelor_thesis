#!/usr/bin/env python3
"""
Create compelling visualizations of inference benchmark results for thesis.
Generates publication-quality plots showing performance advantages of sparse attention.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Your benchmark results
benchmark_data = {
    'Model': ['Dense', 'Dense', 'Sparse', 'Sparse'],
    'Batch': [1, 8, 1, 8],
    'Latency_ms': [33.94, 258.34, 28.49, 121.42],
    'Throughput_tok_s': [30170, 31711, 35939, 67469],
    'GPU_Util_pct': [92.2, 99.2, 68.1, 91.4],
    'Peak_Mem_MB': [2336, 3407, 1946, 2170],
    'Parameters_M': [63.7, 63.7, 37.9, 37.9]
}

df = pd.DataFrame(benchmark_data)

def create_latency_comparison():
    """Create latency comparison plot showing sparse advantage."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Prepare data
    batch_sizes = [1, 8]
    dense_latency = [33.94, 258.34]
    sparse_latency = [28.49, 121.42]
    
    x = np.arange(len(batch_sizes))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, dense_latency, width, label='Dense', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, sparse_latency, width, label='Sparse', color='#2ecc71', alpha=0.8)
    
    # Add improvement percentages
    improvements = [(dense_latency[i] - sparse_latency[i]) / dense_latency[i] * 100 
                   for i in range(len(batch_sizes))]
    
    for i, (bar1, bar2, improvement) in enumerate(zip(bars1, bars2, improvements)):
        height = max(bar1.get_height(), bar2.get_height())
        ax.annotate(f'-{improvement:.1f}%', 
                   xy=(i, height + height * 0.05), 
                   ha='center', va='bottom', fontweight='bold', color='green', fontsize=12)
    
    # Add value labels on bars
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}ms',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Batch Size', fontsize=14, fontweight='bold')
    ax.set_ylabel('Latency per Sequence (ms)', fontsize=14, fontweight='bold')
    ax.set_title('Inference Latency Comparison: Sparse vs Dense Attention\n(Lower is Better)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('inference_latency_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('inference_latency_comparison.pdf', bbox_inches='tight')
    return fig

def create_throughput_comparison():
    """Create throughput comparison plot."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Prepare data
    batch_sizes = [1, 8]
    dense_throughput = [30170, 31711]
    sparse_throughput = [35939, 67469]
    
    x = np.arange(len(batch_sizes))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, dense_throughput, width, label='Dense', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, sparse_throughput, width, label='Sparse', color='#2ecc71', alpha=0.8)
    
    # Add improvement percentages
    improvements = [(sparse_throughput[i] - dense_throughput[i]) / dense_throughput[i] * 100 
                   for i in range(len(batch_sizes))]
    
    for i, (bar1, bar2, improvement) in enumerate(zip(bars1, bars2, improvements)):
        height = max(bar1.get_height(), bar2.get_height())
        ax.annotate(f'+{improvement:.1f}%', 
                   xy=(i, height + height * 0.05), 
                   ha='center', va='bottom', fontweight='bold', color='green', fontsize=12)
    
    # Add value labels on bars
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:,.0f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Batch Size', fontsize=14, fontweight='bold')
    ax.set_ylabel('Throughput (tokens/second)', fontsize=14, fontweight='bold')
    ax.set_title('Inference Throughput Comparison: Sparse vs Dense Attention\n(Higher is Better)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Format y-axis with thousands separator
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    plt.tight_layout()
    plt.savefig('inference_throughput_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('inference_throughput_comparison.pdf', bbox_inches='tight')
    return fig

def create_memory_efficiency_plot():
    """Create memory usage and parameter efficiency plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Memory usage comparison
    batch_sizes = [1, 8]
    dense_memory = [2336, 3407]
    sparse_memory = [1946, 2170]
    
    x = np.arange(len(batch_sizes))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, dense_memory, width, label='Dense', color='#e74c3c', alpha=0.8)
    bars2 = ax1.bar(x + width/2, sparse_memory, width, label='Sparse', color='#2ecc71', alpha=0.8)
    
    # Add memory savings percentages
    savings = [(dense_memory[i] - sparse_memory[i]) / dense_memory[i] * 100 
               for i in range(len(batch_sizes))]
    
    for i, (bar1, bar2, saving) in enumerate(zip(bars1, bars2, savings)):
        height = max(bar1.get_height(), bar2.get_height())
        ax1.annotate(f'-{saving:.1f}%', 
                    xy=(i, height + height * 0.05), 
                    ha='center', va='bottom', fontweight='bold', color='green', fontsize=12)
    
    # Add value labels
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.0f}MB',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    ax1.set_xlabel('Batch Size', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Peak Memory Usage (MB)', fontsize=14, fontweight='bold')
    ax1.set_title('Memory Efficiency', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(batch_sizes)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Parameter efficiency (pie chart)
    sizes = [63.7, 37.9]
    labels = ['Dense\n(63.7M params)', 'Sparse\n(37.9M params)']
    colors = ['#e74c3c', '#2ecc71']
    explode = (0, 0.1)  # explode sparse slice
    
    wedges, texts, autotexts = ax2.pie(sizes, explode=explode, labels=labels, colors=colors,
                                      autopct='%1.1f%%', shadow=True, startangle=90,
                                      textprops={'fontsize': 12})
    
    # Add parameter reduction annotation
    ax2.annotate('40.4% fewer\nparameters', xy=(0.5, -0.3), xycoords='axes fraction',
                ha='center', va='center', fontsize=14, fontweight='bold', color='green',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    
    ax2.set_title('Model Size Comparison', fontsize=14, fontweight='bold')
    
    plt.suptitle('Resource Efficiency: Memory Usage and Model Size', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('inference_memory_efficiency.png', dpi=300, bbox_inches='tight')
    plt.savefig('inference_memory_efficiency.pdf', bbox_inches='tight')
    return fig

def create_gpu_utilization_plot():
    """Create GPU utilization comparison."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    batch_sizes = [1, 8]
    dense_util = [92.2, 99.2]
    sparse_util = [68.1, 91.4]
    
    # Create line plot
    ax.plot(batch_sizes, dense_util, 'o-', color='#e74c3c', linewidth=3, markersize=8, 
            label='Dense', alpha=0.8)
    ax.plot(batch_sizes, sparse_util, 's-', color='#2ecc71', linewidth=3, markersize=8, 
            label='Sparse', alpha=0.8)
    
    # Add value annotations
    for i, (batch, dense, sparse) in enumerate(zip(batch_sizes, dense_util, sparse_util)):
        ax.annotate(f'{dense:.1f}%', (batch, dense), xytext=(5, 10), 
                   textcoords='offset points', fontsize=11, fontweight='bold')
        ax.annotate(f'{sparse:.1f}%', (batch, sparse), xytext=(5, -15), 
                   textcoords='offset points', fontsize=11, fontweight='bold')
    
    # Add saturation line
    ax.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Hardware Limit')
    
    ax.set_xlabel('Batch Size', fontsize=14, fontweight='bold')
    ax.set_ylabel('GPU Utilization (%)', fontsize=14, fontweight='bold')
    ax.set_title('GPU Utilization Patterns: Scaling Characteristics', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(batch_sizes)
    ax.set_ylim(60, 105)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add interpretation text
    ax.text(0.02, 0.98, 'Sparse attention shows\nmore efficient scaling\nwith headroom for\nlarger batches', 
            transform=ax.transAxes, fontsize=11, va='top', ha='left',
            bbox=dict(boxstyle="round,pad=0.4", facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('inference_gpu_utilization.png', dpi=300, bbox_inches='tight')
    plt.savefig('inference_gpu_utilization.pdf', bbox_inches='tight')
    return fig

def create_combined_performance_dashboard():
    """Create a comprehensive dashboard showing all metrics."""
    fig = plt.figure(figsize=(16, 12))
    
    # Create subplots
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Latency subplot
    ax1 = fig.add_subplot(gs[0, 0])
    batch_sizes = [1, 8]
    dense_latency = [33.94, 258.34]
    sparse_latency = [28.49, 121.42]
    x = np.arange(len(batch_sizes))
    width = 0.35
    bars1 = ax1.bar(x - width/2, dense_latency, width, label='Dense', color='#e74c3c', alpha=0.8)
    bars2 = ax1.bar(x + width/2, sparse_latency, width, label='Sparse', color='#2ecc71', alpha=0.8)
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title('(A) Latency Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(batch_sizes)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Throughput subplot
    ax2 = fig.add_subplot(gs[0, 1])
    dense_throughput = [30170, 31711]
    sparse_throughput = [35939, 67469]
    bars3 = ax2.bar(x - width/2, dense_throughput, width, label='Dense', color='#e74c3c', alpha=0.8)
    bars4 = ax2.bar(x + width/2, sparse_throughput, width, label='Sparse', color='#2ecc71', alpha=0.8)
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Throughput (tok/s)')
    ax2.set_title('(B) Throughput Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(batch_sizes)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))
    
    # Memory usage subplot
    ax3 = fig.add_subplot(gs[1, 0])
    dense_memory = [2336, 3407]
    sparse_memory = [1946, 2170]
    bars5 = ax3.bar(x - width/2, dense_memory, width, label='Dense', color='#e74c3c', alpha=0.8)
    bars6 = ax3.bar(x + width/2, sparse_memory, width, label='Sparse', color='#2ecc71', alpha=0.8)
    ax3.set_xlabel('Batch Size')
    ax3.set_ylabel('Memory (MB)')
    ax3.set_title('(C) Memory Usage')
    ax3.set_xticks(x)
    ax3.set_xticklabels(batch_sizes)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # GPU utilization subplot
    ax4 = fig.add_subplot(gs[1, 1])
    dense_util = [92.2, 99.2]
    sparse_util = [68.1, 91.4]
    ax4.plot(batch_sizes, dense_util, 'o-', color='#e74c3c', linewidth=3, markersize=8, label='Dense')
    ax4.plot(batch_sizes, sparse_util, 's-', color='#2ecc71', linewidth=3, markersize=8, label='Sparse')
    ax4.axhline(y=100, color='red', linestyle='--', alpha=0.7)
    ax4.set_xlabel('Batch Size')
    ax4.set_ylabel('GPU Utilization (%)')
    ax4.set_title('(D) GPU Utilization')
    ax4.set_xticks(batch_sizes)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Performance summary table
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    # Create summary table
    improvements = [
        ['Metric', 'Batch=1', 'Batch=8', 'Advantage'],
        ['Latency', '-16.1%', '-53.0%', 'Faster'],
        ['Throughput', '+19.1%', '+112.8%', 'Higher'],
        ['Memory', '-16.7%', '-36.3%', 'Lower'],
        ['Parameters', '-40.4%', '-40.4%', 'Smaller']
    ]
    
    table = ax5.table(cellText=improvements[1:], colLabels=improvements[0],
                     cellLoc='center', loc='center', colWidths=[0.2, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(improvements)):
        for j in range(len(improvements[0])):
            cell = table[(i, j)] if i == 0 else table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#34495e')
                cell.set_text_props(weight='bold', color='white')
            elif j == 3:  # Advantage column
                cell.set_facecolor('#2ecc71')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#ecf0f1')
    
    ax5.set_title('(E) Performance Summary: Sparse vs Dense Attention', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.suptitle('Inference Performance Analysis: Comprehensive Comparison', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig('inference_performance_dashboard.png', dpi=300, bbox_inches='tight')
    plt.savefig('inference_performance_dashboard.pdf', bbox_inches='tight')
    return fig

def create_efficiency_ratio_plot():
    """Create performance per parameter and performance per memory plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Performance per parameter (throughput/parameters)
    batch_sizes = [1, 8]
    dense_perf_per_param = [30170/63.7, 31711/63.7]  # tok/s per million params
    sparse_perf_per_param = [35939/37.9, 67469/37.9]
    
    x = np.arange(len(batch_sizes))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, dense_perf_per_param, width, label='Dense', color='#e74c3c', alpha=0.8)
    bars2 = ax1.bar(x + width/2, sparse_perf_per_param, width, label='Sparse', color='#2ecc71', alpha=0.8)
    
    # Add efficiency advantage percentages
    efficiency_gains = [(sparse_perf_per_param[i] - dense_perf_per_param[i]) / dense_perf_per_param[i] * 100 
                       for i in range(len(batch_sizes))]
    
    for i, (bar1, bar2, gain) in enumerate(zip(bars1, bars2, efficiency_gains)):
        height = max(bar1.get_height(), bar2.get_height())
        ax1.annotate(f'+{gain:.0f}%', 
                    xy=(i, height + height * 0.05), 
                    ha='center', va='bottom', fontweight='bold', color='green', fontsize=12)
    
    ax1.set_xlabel('Batch Size', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Tokens/sec per Million Parameters', fontsize=14, fontweight='bold')
    ax1.set_title('Parameter Efficiency', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(batch_sizes)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Performance per memory (throughput/memory)
    dense_perf_per_mem = [30170/2336, 31711/3407]  # tok/s per MB
    sparse_perf_per_mem = [35939/1946, 67469/2170]
    
    bars3 = ax2.bar(x - width/2, dense_perf_per_mem, width, label='Dense', color='#e74c3c', alpha=0.8)
    bars4 = ax2.bar(x + width/2, sparse_perf_per_mem, width, label='Sparse', color='#2ecc71', alpha=0.8)
    
    # Add memory efficiency advantage percentages
    mem_efficiency_gains = [(sparse_perf_per_mem[i] - dense_perf_per_mem[i]) / dense_perf_per_mem[i] * 100 
                           for i in range(len(batch_sizes))]
    
    for i, (bar3, bar4, gain) in enumerate(zip(bars3, bars4, mem_efficiency_gains)):
        height = max(bar3.get_height(), bar4.get_height())
        ax2.annotate(f'+{gain:.0f}%', 
                    xy=(i, height + height * 0.05), 
                    ha='center', va='bottom', fontweight='bold', color='green', fontsize=12)
    
    ax2.set_xlabel('Batch Size', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Tokens/sec per MB Memory', fontsize=14, fontweight='bold')
    ax2.set_title('Memory Efficiency', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(batch_sizes)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Resource Efficiency Analysis: Performance Density', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('inference_efficiency_ratios.png', dpi=300, bbox_inches='tight')
    plt.savefig('inference_efficiency_ratios.pdf', bbox_inches='tight')
    return fig

def main():
    """Generate all inference performance plots."""
    print("🎨 Creating inference performance visualizations...")
    
    # Create output directory
    output_dir = Path('inference_plots')
    output_dir.mkdir(exist_ok=True)
    
    # Change to output directory
    import os
    os.chdir(output_dir)
    
    # Generate all plots
    plots = []
    
    print("  📊 Creating latency comparison plot...")
    plots.append(create_latency_comparison())
    
    print("  📈 Creating throughput comparison plot...")
    plots.append(create_throughput_comparison())
    
    print("  💾 Creating memory efficiency plot...")
    plots.append(create_memory_efficiency_plot())
    
    print("  🖥️ Creating GPU utilization plot...")
    plots.append(create_gpu_utilization_plot())
    
    print("  📋 Creating comprehensive dashboard...")
    plots.append(create_combined_performance_dashboard())
    
    print("  ⚡ Creating efficiency ratio plot...")
    plots.append(create_efficiency_ratio_plot())
    
    # Show plots
    plt.show()
    
    print(f"\n✅ All plots saved to: {output_dir.absolute()}")
    print("📁 Files generated:")
    print("  - inference_latency_comparison.png/.pdf")
    print("  - inference_throughput_comparison.png/.pdf") 
    print("  - inference_memory_efficiency.png/.pdf")
    print("  - inference_gpu_utilization.png/.pdf")
    print("  - inference_performance_dashboard.png/.pdf")
    print("  - inference_efficiency_ratios.png/.pdf")
    
    print("\n🎯 For your thesis:")
    print("  • Use the dashboard for comprehensive overview")
    print("  • Individual plots for focused discussions")
    print("  • All plots are publication-ready (300 DPI)")
    print("  • PDF versions included for LaTeX")

if __name__ == "__main__":
    main() 