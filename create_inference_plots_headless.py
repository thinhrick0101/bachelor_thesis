#!/usr/bin/env python3
"""
Create compelling visualizations of inference benchmark results for thesis.
Generates publication-quality plots showing performance advantages of sparse attention.
Headless version that works without display.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
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
    plt.close(fig)
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
    plt.close(fig)
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
    plt.close(fig)
    return fig

def create_combined_performance_dashboard():
    """Create a comprehensive dashboard showing all metrics."""
    fig = plt.figure(figsize=(16, 10))
    
    # Create subplots with more space
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Latency subplot
    ax1 = fig.add_subplot(gs[0, 0])
    batch_sizes = [1, 8]
    dense_latency = [33.94, 258.34]
    sparse_latency = [28.49, 121.42]
    x = np.arange(len(batch_sizes))
    width = 0.35
    bars1 = ax1.bar(x - width/2, dense_latency, width, label='Dense', color='#e74c3c', alpha=0.8)
    bars2 = ax1.bar(x + width/2, sparse_latency, width, label='Sparse', color='#2ecc71', alpha=0.8)
    
    # Add improvement percentages
    improvements = [(dense_latency[i] - sparse_latency[i]) / dense_latency[i] * 100 
                   for i in range(len(batch_sizes))]
    for i, improvement in enumerate(improvements):
        ax1.text(i, max(dense_latency[i], sparse_latency[i]) * 1.1, f'-{improvement:.1f}%', 
                ha='center', va='bottom', fontweight='bold', color='green', fontsize=11)
    
    ax1.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('(A) Latency Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(batch_sizes)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Throughput subplot
    ax2 = fig.add_subplot(gs[0, 1])
    dense_throughput = [30170, 31711]
    sparse_throughput = [35939, 67469]
    bars3 = ax2.bar(x - width/2, dense_throughput, width, label='Dense', color='#e74c3c', alpha=0.8)
    bars4 = ax2.bar(x + width/2, sparse_throughput, width, label='Sparse', color='#2ecc71', alpha=0.8)
    
    # Add improvement percentages
    throughput_improvements = [(sparse_throughput[i] - dense_throughput[i]) / dense_throughput[i] * 100 
                              for i in range(len(batch_sizes))]
    for i, improvement in enumerate(throughput_improvements):
        ax2.text(i, max(dense_throughput[i], sparse_throughput[i]) * 1.1, f'+{improvement:.1f}%', 
                ha='center', va='bottom', fontweight='bold', color='green', fontsize=11)
    
    ax2.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Throughput (tokens/s)', fontsize=12, fontweight='bold')
    ax2.set_title('(B) Throughput Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(batch_sizes)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))
    
    # Memory usage subplot
    ax3 = fig.add_subplot(gs[1, 0])
    dense_memory = [2336, 3407]
    sparse_memory = [1946, 2170]
    bars5 = ax3.bar(x - width/2, dense_memory, width, label='Dense', color='#e74c3c', alpha=0.8)
    bars6 = ax3.bar(x + width/2, sparse_memory, width, label='Sparse', color='#2ecc71', alpha=0.8)
    
    # Add memory savings percentages
    memory_savings = [(dense_memory[i] - sparse_memory[i]) / dense_memory[i] * 100 
                     for i in range(len(batch_sizes))]
    for i, saving in enumerate(memory_savings):
        ax3.text(i, max(dense_memory[i], sparse_memory[i]) * 1.1, f'-{saving:.1f}%', 
                ha='center', va='bottom', fontweight='bold', color='green', fontsize=11)
    
    ax3.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Memory (MB)', fontsize=12, fontweight='bold')
    ax3.set_title('(C) Memory Usage', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(batch_sizes)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # GPU utilization subplot
    ax4 = fig.add_subplot(gs[1, 1])
    dense_util = [92.2, 99.2]
    sparse_util = [68.1, 91.4]
    ax4.plot(batch_sizes, dense_util, 'o-', color='#e74c3c', linewidth=3, markersize=8, label='Dense')
    ax4.plot(batch_sizes, sparse_util, 's-', color='#2ecc71', linewidth=3, markersize=8, label='Sparse')
    ax4.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Hardware Limit')
    
    # Add value annotations
    for i, (batch, dense, sparse) in enumerate(zip(batch_sizes, dense_util, sparse_util)):
        ax4.annotate(f'{dense:.1f}%', (batch, dense), xytext=(5, 10), 
                   textcoords='offset points', fontsize=10, fontweight='bold')
        ax4.annotate(f'{sparse:.1f}%', (batch, sparse), xytext=(5, -15), 
                   textcoords='offset points', fontsize=10, fontweight='bold')
    
    ax4.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax4.set_ylabel('GPU Utilization (%)', fontsize=12, fontweight='bold')
    ax4.set_title('(D) GPU Utilization', fontsize=14, fontweight='bold')
    ax4.set_xticks(batch_sizes)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(60, 105)
    
    plt.suptitle('Inference Performance Analysis: Comprehensive Comparison', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig('inference_performance_dashboard.png', dpi=300, bbox_inches='tight')
    plt.savefig('inference_performance_dashboard.pdf', bbox_inches='tight')
    plt.close(fig)
    return fig

def create_efficiency_summary():
    """Create a summary plot showing key efficiency metrics."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Metrics data
    metrics = ['Latency\n(Batch=1)', 'Latency\n(Batch=8)', 'Throughput\n(Batch=1)', 
               'Throughput\n(Batch=8)', 'Memory\n(Batch=1)', 'Memory\n(Batch=8)', 'Parameters']
    improvements = [-16.1, -53.0, +19.1, +112.8, -16.7, -36.3, -40.4]
    colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in improvements]
    
    # Create horizontal bar chart
    bars = ax.barh(metrics, improvements, color=colors, alpha=0.7)
    
    # Add value labels
    for i, (bar, improvement) in enumerate(zip(bars, improvements)):
        width = bar.get_width()
        label_x = width + 2 if width > 0 else width - 2
        ha = 'left' if width > 0 else 'right'
        ax.text(label_x, bar.get_y() + bar.get_height()/2, f'{improvement:+.1f}%', 
                ha=ha, va='center', fontweight='bold', fontsize=12)
    
    # Add vertical line at zero
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Customize
    ax.set_xlabel('Improvement (%)', fontsize=14, fontweight='bold')
    ax.set_title('Sparse vs Dense Attention: Performance Summary\n(Positive = Better for Sparse)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', alpha=0.7, label='Sparse Advantage'),
                      Patch(facecolor='red', alpha=0.7, label='Dense Better')]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=12)
    
    # Set x-axis limits
    ax.set_xlim(-60, 120)
    
    plt.tight_layout()
    plt.savefig('inference_efficiency_summary.png', dpi=300, bbox_inches='tight')
    plt.savefig('inference_efficiency_summary.pdf', bbox_inches='tight')
    plt.close(fig)
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
    print("  📊 Creating latency comparison plot...")
    create_latency_comparison()
    
    print("  📈 Creating throughput comparison plot...")
    create_throughput_comparison()
    
    print("  💾 Creating memory efficiency plot...")
    create_memory_efficiency_plot()
    
    print("  📋 Creating comprehensive dashboard...")
    create_combined_performance_dashboard()
    
    print("  ⚡ Creating efficiency summary plot...")
    create_efficiency_summary()
    
    print(f"\n✅ All plots saved to: {output_dir.absolute()}")
    print("📁 Files generated:")
    for file in sorted(output_dir.glob("*.png")):
        print(f"  - {file.name}")
    
    print("\n🎯 Thesis Integration Tips:")
    print("  • Use 'inference_performance_dashboard.png' for Section 6.4 overview")
    print("  • Use 'inference_efficiency_summary.png' for quick comparison")
    print("  • Individual plots for detailed analysis")
    print("  • All plots are 300 DPI publication-ready")
    print("  • PDF versions included for LaTeX documents")
    
    print("\n📊 Key Visual Messages:")
    print("  ✓ Sparse attention is 16-53% faster (latency)")
    print("  ✓ Sparse attention has 19-113% higher throughput")
    print("  ✓ Sparse attention uses 17-36% less memory")
    print("  ✓ Sparse attention has 40% fewer parameters")

if __name__ == "__main__":
    main() 