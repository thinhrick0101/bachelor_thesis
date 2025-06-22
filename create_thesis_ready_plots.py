#!/usr/bin/env python3
"""
Create Thesis-Ready Plots with Professional Titles
Clean, publication-quality plots for Track B normalized entropy analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def create_thesis_ready_plots():
    """Generate publication-quality plots with clean, professional titles"""
    print("🎨 Creating thesis-ready plots with professional formatting...")
    
    # Load the updated metrics
    df = pd.read_csv('head_metrics.csv')
    print(f"📊 Loaded {len(df)} head metrics")
    
    # Set publication-quality style
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'grid.alpha': 0.3,
        'figure.dpi': 300
    })
    
    # Create figure with clean, professional layout
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Attention Head Analysis: Normalized Entropy Metrics', 
                 fontsize=18, fontweight='bold', y=0.95)
    
    # Plot 1: Entropy distribution by layer
    ax1 = axes[0, 0]
    layers = sorted(df['layer'].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, len(layers)))
    
    for i, layer in enumerate(layers):
        layer_data = df[df['layer'] == layer]['entropy']
        ax1.hist(layer_data, alpha=0.7, label=f'L{layer}', bins=12, 
                color=colors[i], edgecolor='white', linewidth=0.5)
    
    ax1.set_xlabel('Normalised entropy (0–1)', fontweight='bold')
    ax1.set_ylabel('Count', fontweight='bold')
    ax1.set_title('(A) Distribution by Layer', fontweight='bold', pad=15)
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, 
              ncol=2, frameon=False)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.1, 0.8)
    
    # Plot 2: Entropy vs Distance scatter
    ax2 = axes[0, 1]
    scatter = ax2.scatter(df['entropy'], df['distance'], 
                         c=df['layer'], cmap='viridis', alpha=0.8, s=60,
                         edgecolors='white', linewidth=0.5)
    ax2.set_xlabel('Normalised entropy (0–1)', fontweight='bold')
    ax2.set_ylabel('Average distance (tokens)', fontweight='bold')
    ax2.set_title('(B) Entropy vs. Token Distance', fontweight='bold', pad=15)
    cbar = plt.colorbar(scatter, ax=ax2, shrink=0.8)
    cbar.set_label('Layer', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.1, 0.8)
    
    # Plot 3: Entropy vs Sparsity
    ax3 = axes[1, 0]
    scatter2 = ax3.scatter(df['entropy'], df['sparsity'], 
                          c=df['layer'], cmap='plasma', alpha=0.8, s=60,
                          edgecolors='white', linewidth=0.5)
    ax3.set_xlabel('Normalised entropy (0–1)', fontweight='bold')
    ax3.set_ylabel('Sparsity', fontweight='bold')
    ax3.set_title('(C) Entropy vs. Sparsity', fontweight='bold', pad=15)
    cbar2 = plt.colorbar(scatter2, ax=ax3, shrink=0.8)
    cbar2.set_label('Layer', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0.1, 0.8)
    ax3.set_ylim(0.85, 1.0)
    
    # Plot 4: Layer-wise averages with error bars
    ax4 = axes[1, 1]
    layer_means = df.groupby('layer')['entropy'].mean()
    layer_stds = df.groupby('layer')['entropy'].std()
    
    x_pos = range(len(layer_means))
    bars = ax4.bar(x_pos, layer_means, yerr=layer_stds, 
                   capsize=4, alpha=0.8, color='steelblue', 
                   edgecolor='navy', linewidth=1.2, error_kw={'linewidth': 1.5})
    
    ax4.set_xlabel('Layer', fontweight='bold')
    ax4.set_ylabel('Normalised entropy (0–1)', fontweight='bold')
    ax4.set_title('(D) Layer-wise Average', fontweight='bold', pad=15)
    ax4.set_xticks(x_pos[::2])  # Show every other layer for clarity
    ax4.set_xticklabels([f'{i}' for i in layer_means.index[::2]])
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim(0.3, 0.7)
    
    # Add value labels on selected bars
    for i, (bar, mean, std) in enumerate(zip(bars, layer_means, layer_stds)):
        if i % 3 == 0:  # Show labels every 3rd bar to avoid crowding
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                    f'{mean:.2f}', ha='center', va='bottom', 
                    fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Make room for main title
    plt.savefig('thesis_ready_entropy_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Saved thesis_ready_entropy_plots.png")

def create_single_panel_plots():
    """Create individual plots for flexible thesis integration"""
    print("📊 Creating individual panel plots...")
    
    df = pd.read_csv('head_metrics.csv')
    
    # Set consistent style
    plt.rcParams.update({
        'font.size': 14,
        'font.family': 'serif',
        'axes.linewidth': 1.2,
        'figure.dpi': 300
    })
    
    # Panel A: Distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    layers = sorted(df['layer'].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, len(layers)))
    
    for i, layer in enumerate(layers):
        layer_data = df[df['layer'] == layer]['entropy']
        ax.hist(layer_data, alpha=0.7, label=f'Layer {layer}', bins=12, 
               color=colors[i], edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel('Normalised entropy (0–1)', fontweight='bold', fontsize=16)
    ax.set_ylabel('Count', fontweight='bold', fontsize=16)
    ax.set_title('Distribution of Normalised Entropy by Layer', 
                fontweight='bold', fontsize=18, pad=20)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12, 
             ncol=2, frameon=False)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.1, 0.8)
    
    plt.tight_layout()
    plt.savefig('figure_entropy_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Panel B: Scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(df['entropy'], df['distance'], 
                        c=df['layer'], cmap='viridis', alpha=0.8, s=80,
                        edgecolors='white', linewidth=0.8)
    ax.set_xlabel('Normalised entropy (0–1)', fontweight='bold', fontsize=16)
    ax.set_ylabel('Average distance (tokens)', fontweight='bold', fontsize=16)
    ax.set_title('Normalised Entropy vs. Average Token Distance', 
                fontweight='bold', fontsize=18, pad=20)
    cbar = plt.colorbar(scatter, shrink=0.8)
    cbar.set_label('Layer', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.1, 0.8)
    
    plt.tight_layout()
    plt.savefig('figure_entropy_vs_distance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Saved individual panel plots")

def create_comparison_with_clean_title():
    """Create before/after comparison with professional title"""
    print("🔄 Creating clean comparison plot...")
    
    df = pd.read_csv('head_metrics.csv')
    
    plt.rcParams.update({
        'font.size': 14,
        'font.family': 'serif',
        'figure.dpi': 300
    })
    
    fig, (ax_old, ax_new) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Track B Label Update: Entropy Normalization', 
                fontsize=18, fontweight='bold')
    
    # Old style (before Track B)
    ax_old.scatter(df['entropy'], df['distance'], alpha=0.7, color='#d62728', s=60)
    ax_old.set_xlabel('Entropy (bits)', fontweight='bold', color='#d62728', fontsize=14)
    ax_old.set_ylabel('Average distance (tokens)', fontweight='bold', fontsize=14)
    ax_old.set_title('Before: Unclear Base', fontweight='bold', color='#d62728', fontsize=16)
    ax_old.grid(True, alpha=0.3)
    ax_old.text(0.05, 0.95, '❌ Ambiguous\nentropy base', transform=ax_old.transAxes,
               fontsize=12, fontweight='bold', color='#d62728', 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # New style (after Track B)
    ax_new.scatter(df['entropy'], df['distance'], alpha=0.7, color='#2ca02c', s=60)
    ax_new.set_xlabel('Normalised entropy (0–1)', fontweight='bold', color='#2ca02c', fontsize=14)
    ax_new.set_ylabel('Average distance (tokens)', fontweight='bold', fontsize=14)
    ax_new.set_title('After: Clear Definition', fontweight='bold', color='#2ca02c', fontsize=16)
    ax_new.grid(True, alpha=0.3)
    ax_new.text(0.05, 0.95, '✅ H_norm = H_nat / ln(L)\nStandard definition', 
               transform=ax_new.transAxes, fontsize=12, fontweight='bold', color='#2ca02c',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('track_b_comparison_clean.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Saved track_b_comparison_clean.png")

def main():
    print("🎯 Creating Thesis-Ready Plots with Professional Titles")
    print("=" * 60)
    
    # Create main multi-panel figure
    create_thesis_ready_plots()
    
    # Create individual panels for flexible use
    create_single_panel_plots()
    
    # Create clean comparison
    create_comparison_with_clean_title()
    
    print("\n🎯 Thesis-Ready Plots Complete!")
    print("📊 Generated Files:")
    print("   📈 thesis_ready_entropy_plots.png - Main 4-panel figure")
    print("   📈 figure_entropy_distribution.png - Individual distribution plot")
    print("   📈 figure_entropy_vs_distance.png - Individual scatter plot")
    print("   📈 track_b_comparison_clean.png - Clean before/after comparison")
    print("\n📝 All plots use professional formatting with:")
    print("   ✅ Clear 'Normalised entropy (0–1)' labels")
    print("   ✅ Publication-quality typography")
    print("   ✅ Clean panel labels (A), (B), (C), (D)")
    print("   ✅ Proper color schemes and grid styling")
    print("   ✅ High-resolution (300 DPI) for thesis integration")

if __name__ == "__main__":
    main() 