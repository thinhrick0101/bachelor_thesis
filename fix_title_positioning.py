#!/usr/bin/env python3
"""
Fix Title Positioning for Thesis-Ready Plots
Adjusts spacing and layout to prevent title overlap with subplots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_fixed_thesis_plots():
    """Generate thesis plots with properly positioned main title"""
    print("🔧 Fixing title positioning in thesis-ready plots...")
    
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
    
    # Create figure with proper spacing for title
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Add main title with proper positioning
    fig.suptitle('Attention Head Analysis: Normalized Entropy Metrics', 
                 fontsize=18, fontweight='bold', y=0.98)  # Move higher
    
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
    ax1.set_title('(A) Distribution by Layer', fontweight='bold', pad=20)  # More padding
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
    ax2.set_title('(B) Entropy vs. Token Distance', fontweight='bold', pad=20)  # More padding
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
    ax3.set_title('(C) Entropy vs. Sparsity', fontweight='bold', pad=20)  # More padding
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
    ax4.set_title('(D) Layer-wise Average', fontweight='bold', pad=20)  # More padding
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
    
    # Adjust layout with more top margin for title
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)  # Leave more room for main title
    
    # Save with fixed positioning
    plt.savefig('thesis_ready_entropy_plots_fixed.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Saved thesis_ready_entropy_plots_fixed.png")

def create_alternative_layouts():
    """Create alternative layouts with different title approaches"""
    print("🎨 Creating alternative layout options...")
    
    df = pd.read_csv('head_metrics.csv')
    
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.linewidth': 1.2,
        'figure.dpi': 300
    })
    
    # Option 1: No main title, just panel titles
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot setup (same as before but without main title)
    ax1 = axes[0, 0]
    layers = sorted(df['layer'].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, len(layers)))
    
    for i, layer in enumerate(layers):
        layer_data = df[df['layer'] == layer]['entropy']
        ax1.hist(layer_data, alpha=0.7, label=f'L{layer}', bins=12, 
                color=colors[i], edgecolor='white', linewidth=0.5)
    
    ax1.set_xlabel('Normalised entropy (0–1)', fontweight='bold')
    ax1.set_ylabel('Count', fontweight='bold')
    ax1.set_title('(A) Distribution of Normalised Entropy by Layer', fontweight='bold', fontsize=14)
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, 
              ncol=2, frameon=False)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.1, 0.8)
    
    # Other plots...
    ax2 = axes[0, 1]
    scatter = ax2.scatter(df['entropy'], df['distance'], 
                         c=df['layer'], cmap='viridis', alpha=0.8, s=60,
                         edgecolors='white', linewidth=0.5)
    ax2.set_xlabel('Normalised entropy (0–1)', fontweight='bold')
    ax2.set_ylabel('Average distance (tokens)', fontweight='bold')
    ax2.set_title('(B) Normalised Entropy vs. Token Distance', fontweight='bold', fontsize=14)
    cbar = plt.colorbar(scatter, ax=ax2, shrink=0.8)
    cbar.set_label('Layer', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.1, 0.8)
    
    ax3 = axes[1, 0]
    scatter2 = ax3.scatter(df['entropy'], df['sparsity'], 
                          c=df['layer'], cmap='plasma', alpha=0.8, s=60,
                          edgecolors='white', linewidth=0.5)
    ax3.set_xlabel('Normalised entropy (0–1)', fontweight='bold')
    ax3.set_ylabel('Sparsity', fontweight='bold')
    ax3.set_title('(C) Normalised Entropy vs. Sparsity', fontweight='bold', fontsize=14)
    cbar2 = plt.colorbar(scatter2, ax=ax3, shrink=0.8)
    cbar2.set_label('Layer', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0.1, 0.8)
    ax3.set_ylim(0.85, 1.0)
    
    ax4 = axes[1, 1]
    layer_means = df.groupby('layer')['entropy'].mean()
    layer_stds = df.groupby('layer')['entropy'].std()
    x_pos = range(len(layer_means))
    bars = ax4.bar(x_pos, layer_means, yerr=layer_stds, 
                   capsize=4, alpha=0.8, color='steelblue', 
                   edgecolor='navy', linewidth=1.2, error_kw={'linewidth': 1.5})
    
    ax4.set_xlabel('Layer', fontweight='bold')
    ax4.set_ylabel('Normalised entropy (0–1)', fontweight='bold')
    ax4.set_title('(D) Average Normalised Entropy by Layer', fontweight='bold', fontsize=14)
    ax4.set_xticks(x_pos[::2])
    ax4.set_xticklabels([f'{i}' for i in layer_means.index[::2]])
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim(0.3, 0.7)
    
    plt.tight_layout()
    plt.savefig('thesis_plots_no_main_title.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Saved thesis_plots_no_main_title.png")
    
    # Option 2: Main title with larger figure height
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))  # Taller figure
    
    # Add main title with more space
    fig.suptitle('Attention Head Analysis: Normalized Entropy Metrics', 
                 fontsize=20, fontweight='bold', y=0.96)
    
    # Same plots but with adjusted spacing
    ax1 = axes[0, 0]
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
    
    # Repeat for other subplots...
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
    
    ax4 = axes[1, 1]
    bars = ax4.bar(x_pos, layer_means, yerr=layer_stds, 
                   capsize=4, alpha=0.8, color='steelblue', 
                   edgecolor='navy', linewidth=1.2, error_kw={'linewidth': 1.5})
    ax4.set_xlabel('Layer', fontweight='bold')
    ax4.set_ylabel('Normalised entropy (0–1)', fontweight='bold')
    ax4.set_title('(D) Layer-wise Average', fontweight='bold', pad=15)
    ax4.set_xticks(x_pos[::2])
    ax4.set_xticklabels([f'{i}' for i in layer_means.index[::2]])
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim(0.3, 0.7)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.91)  # Adjust for larger title
    plt.savefig('thesis_plots_tall_format.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Saved thesis_plots_tall_format.png")

def main():
    print("🔧 Fixing Title Positioning Issues")
    print("=" * 40)
    
    # Create fixed version
    create_fixed_thesis_plots()
    
    # Create alternative layouts
    create_alternative_layouts()
    
    print("\n🎯 Title Positioning Fixed!")
    print("📊 Generated Options:")
    print("   📈 thesis_ready_entropy_plots_fixed.png - Fixed spacing, main title")
    print("   📈 thesis_plots_no_main_title.png - No main title, descriptive panel titles")
    print("   📈 thesis_plots_tall_format.png - Taller format with main title")
    print("\n📝 Choose the version that works best for your thesis layout!")

if __name__ == "__main__":
    main() 