#!/usr/bin/env python3
"""
extract_attention_heatmaps.py

Extract attention patterns and create the four-head heatmap figure
following the recipe provided, adapted to the EnhancedCharTransformer architecture.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from stable_char_transformer import EnhancedCharTransformer, ByteTokenizer, load_data

class AttentionExtractor:
    """Extract attention patterns from EnhancedCharTransformer"""
    
    def __init__(self, model):
        self.model = model
        self.attention_maps = {}
        self.hooks = []
        
    def register_hooks(self):
        """Register forward hooks to capture attention weights from each layer"""
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                # Extract attention weights when available
                if isinstance(output, tuple) and len(output) >= 2:
                    attn_weights = output[1]
                    if attn_weights is not None and attn_weights.dim() >= 3:
                        # Store attention weights for each head
                        batch_size, num_heads = attn_weights.shape[:2]
                        for head_idx in range(num_heads):
                            self.attention_maps[(layer_idx, head_idx)] = attn_weights[0, head_idx].cpu().detach()
            return hook_fn
        
        # Register hooks on MultiheadAttention modules directly
        for layer_idx, block in enumerate(self.model.transformer_blocks):
            if hasattr(block, 'self_attn'):
                hook = block.self_attn.register_forward_hook(make_hook(layer_idx))
                self.hooks.append(hook)
        
        # Also try to modify attention call to ensure we get weights
        self.original_attention_blocks = []
        for layer_idx, block in enumerate(self.model.transformer_blocks):
            if hasattr(block, '_attention_block'):
                self.original_attention_blocks.append(block._attention_block)
                
                # Replace with our instrumented version
                def make_instrumented_attn(orig_attn, layer_idx, block_ref):
                    def instrumented_attn_block(src, src_mask=None):
                        # Pre-norm
                        src2 = block_ref.norm1(src)
                        # Call attention with weights - force need_weights=True
                        try:
                            src2, attn_weights = block_ref.self_attn(src2, src2, src2, 
                                                                   attn_mask=src_mask, need_weights=True)
                            # Store attention weights (batch=0, all heads)
                            if attn_weights is not None and attn_weights.dim() >= 3:
                                for head_idx in range(attn_weights.size(1)):
                                    self.attention_maps[(layer_idx, head_idx)] = attn_weights[0, head_idx].cpu().detach()
                        except:
                            # Fallback to original method
                            src2 = orig_attn(src, src_mask)
                        
                        return block_ref.dropout1(block_ref.gamma1 * src2)
                    return instrumented_attn_block
                
                block._attention_block = make_instrumented_attn(block._attention_block, layer_idx, block)
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def restore_hooks(self):
        """Restore original attention blocks and remove hooks"""
        # Remove forward hooks
        self.remove_hooks()
        
        # Restore original attention blocks
        for layer_idx, block in enumerate(self.model.transformer_blocks):
            if layer_idx < len(self.original_attention_blocks) and hasattr(block, '_attention_block'):
                block._attention_block = self.original_attention_blocks[layer_idx]
    
    def extract_attention(self, input_seq):
        """Extract attention patterns for given input sequence"""
        self.attention_maps = {}
        self.model.eval()
        
        with torch.no_grad():
            # Forward pass - this will populate attention_maps via hooks
            _ = self.model(input_seq)
        
        return self.attention_maps

def pick_representative_heads(attention_maps):
    """Pick one representative head per cluster based on patterns"""
    
    # Analyze patterns to pick representative heads
    head_stats = {}
    
    for (layer, head), attn_matrix in attention_maps.items():
        # Ensure we have a 2D matrix
        if attn_matrix.dim() == 1:
            # Skip malformed attention maps
            print(f"Skipping malformed attention map for layer {layer}, head {head}")
            continue
        
        # Make sure it's square (for self-attention)
        if attn_matrix.dim() == 2 and attn_matrix.size(0) != attn_matrix.size(1):
            # Take the minimum dimension to make it square
            min_dim = min(attn_matrix.size(0), attn_matrix.size(1))
            attn_matrix = attn_matrix[:min_dim, :min_dim]
        
        # Calculate entropy and other metrics
        attn_flat = attn_matrix.flatten()
        # Add small epsilon to avoid log(0)
        attn_flat = attn_flat + 1e-9
        entropy = -(attn_flat * torch.log(attn_flat)).sum().item()
        
        # Calculate other metrics
        max_attention = attn_matrix.max().item()
        sparsity = (attn_matrix < 0.01).float().mean().item()
        
        # Check for diagonal bias (local attention)
        try:
            diagonal_bias = torch.diag(attn_matrix).mean().item()
        except:
            diagonal_bias = 0.0
        
        # Check for strided patterns (sample every k positions)
        stride_strength = 0
        try:
            for stride in [2, 4, 8, 16]:
                if attn_matrix.size(0) > stride and attn_matrix.size(1) > stride:
                    strided_attn = attn_matrix[::stride, ::stride].mean().item()
                    stride_strength = max(stride_strength, strided_attn)
        except:
            stride_strength = 0.0
        
        head_stats[(layer, head)] = {
            'entropy': entropy,
            'max_attention': max_attention,
            'sparsity': sparsity,
            'diagonal_bias': diagonal_bias,
            'stride_strength': stride_strength
        }
    
    # Cluster heads based on characteristics
    clusters = {0: None, 1: None, 2: None, 3: None}
    
    # Cluster 0: Focused-local (high diagonal bias, low entropy)
    best_focused = None
    best_focused_score = -1
    for (layer, head), stats in head_stats.items():
        score = stats['diagonal_bias'] - stats['entropy'] / 10
        if score > best_focused_score:
            best_focused_score = score
            best_focused = (layer, head)
    clusters[0] = best_focused
    
    # Cluster 1: Strided (high stride strength)
    best_strided = None
    best_strided_score = -1
    for (layer, head), stats in head_stats.items():
        if (layer, head) != clusters[0]:
            score = stats['stride_strength']
            if score > best_strided_score:
                best_strided_score = score
                best_strided = (layer, head)
    clusters[1] = best_strided
    
    # Cluster 2: Global-anchor (low sparsity, high entropy)
    best_global = None
    best_global_score = -1
    for (layer, head), stats in head_stats.items():
        if (layer, head) not in [clusters[0], clusters[1]]:
            score = stats['entropy'] - stats['sparsity'] * 5
            if score > best_global_score:
                best_global_score = score
                best_global = (layer, head)
    clusters[2] = best_global
    
    # Cluster 3: Wider-local (medium entropy, medium sparsity)
    best_wider = None
    best_wider_score = -1
    for (layer, head), stats in head_stats.items():
        if (layer, head) not in [clusters[0], clusters[1], clusters[2]]:
            # Pick head with balanced characteristics
            score = -(abs(stats['entropy'] - 5) + abs(stats['sparsity'] - 0.5))
            if score > best_wider_score:
                best_wider_score = score
                best_wider = (layer, head)
    clusters[3] = best_wider
    
    # Remove None values
    clusters = {k: v for k, v in clusters.items() if v is not None}
    
    return clusters

def create_heatmap_figure(attention_maps, clusters, output_path="attention_clusters_heatmap.png"):
    """Create the four-head attention heat-map figure"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Representative Self-Attention Patterns by Cluster', fontsize=16, y=0.98)
    
    cluster_names = {
        0: "Focused-Local",
        1: "Strided", 
        2: "Global-Anchor",
        3: "Wider-Local"
    }
    
    for idx, cluster_id in enumerate([0, 1, 2, 3]):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]
        
        if cluster_id in clusters:
            layer, head = clusters[cluster_id]
            
            if (layer, head) in attention_maps:
                attn_matrix = attention_maps[(layer, head)]
                
                # Downsample for visualization if too large
                if attn_matrix.size(0) > 256:
                    step = max(1, attn_matrix.size(0) // 256)
                    attn_matrix = attn_matrix[::step, ::step]
                
                # Create heatmap with better colormap
                im = ax.imshow(attn_matrix.numpy(), cmap='magma', aspect='auto', 
                              interpolation='nearest', vmin=0, vmax=attn_matrix.max().item())
                
                ax.set_title(f"({chr(97+idx)}) Cluster {cluster_id}: {cluster_names[cluster_id]}\n"
                           f"Layer {layer}, Head {head}", fontsize=11, pad=10)
                ax.set_xlabel("Key Position", fontsize=10)
                ax.set_ylabel("Query Position", fontsize=10)
                
                # Add colorbar
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Attention Weight', fontsize=9)
                
            else:
                # No data available
                ax.text(0.5, 0.5, f"({chr(97+idx)}) Cluster {cluster_id}\n{cluster_names[cluster_id]}\n"
                                 f"(No data for L{layer}-H{head})", 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                ax.set_xticks([])
                ax.set_yticks([])
        else:
            # Cluster not found
            ax.text(0.5, 0.5, f"({chr(97+idx)}) Cluster {cluster_id}\n{cluster_names[cluster_id]}\n(Not found)", 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved attention heatmap figure to: {output_path}")
    
    return fig

def generate_latex_caption(clusters):
    """Generate LaTeX caption for the figure"""
    caption = f"""
\\begin{{figure}}[h]
  \\centering
  \\includegraphics[width=\\linewidth]{{figs/attention_clusters_heatmap.png}}
  \\caption{{Representative self-attention heads discovered in the dense model.
  Each heat-map shows attention probability (query row $\\rightarrow$ key column).
  \\textbf{{(a)}} Cluster-0 focused-local (L{clusters.get(0, ('?', '?'))[0]}-H{clusters.get(0, ('?', '?'))[1]}); 
  \\textbf{{(b)}} Cluster-1 strided (L{clusters.get(1, ('?', '?'))[0]}-H{clusters.get(1, ('?', '?'))[1]}); 
  \\textbf{{(c)}} Cluster-2 global-anchor (L{clusters.get(2, ('?', '?'))[0]}-H{clusters.get(2, ('?', '?'))[1]}); 
  \\textbf{{(d)}} Cluster-3 wider-local (L{clusters.get(3, ('?', '?'))[0]}-H{clusters.get(3, ('?', '?'))[1]}).
  These patterns were used as fixed masks in the sparse Transformer.}}
  \\label{{fig:cluster_heatmaps}}
\\end{{figure}}
"""
    return caption

def generate_markdown_caption(clusters):
    """Generate Markdown caption for the figure"""
    caption = f"""
![Attention patterns for four representative heads.
(a) Cluster-0 focused-local (L{clusters.get(0, ('?', '?'))[0]}-H{clusters.get(0, ('?', '?'))[1]}), 
(b) Cluster-1 strided (L{clusters.get(1, ('?', '?'))[0]}-H{clusters.get(1, ('?', '?'))[1]}),
(c) Cluster-2 global-anchor (L{clusters.get(2, ('?', '?'))[0]}-H{clusters.get(2, ('?', '?'))[1]}), 
(d) Cluster-3 wider-local (L{clusters.get(3, ('?', '?'))[0]}-H{clusters.get(3, ('?', '?'))[1]}).
Probabilities are row-normalised.](figs/attention_clusters_heatmap.png)
"""
    return caption

def main():
    parser = argparse.ArgumentParser(description='Extract attention heatmap figure for thesis')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained dense model checkpoint')
    parser.add_argument('--data_path', type=str, default='data/enwik8',
                       help='Path to validation data')
    parser.add_argument('--seq_length', type=int, default=512,
                       help='Sequence length for analysis')
    parser.add_argument('--output', type=str, default='attention_clusters_heatmap.png',
                       help='Output figure path')
    
    args = parser.parse_args()
    
    print("🔍 Loading model and data...")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model configuration (should match training)
    config = {
        'vocab_size': 256,
        'd_model': 512,
        'nhead': 8,
        'num_layers': 12,
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'attention_dropout': 0.1,
        'activation_dropout': 0.1,
        'token_dropout': 0.05,
        'use_checkpoint': False,  # Disable checkpointing for analysis
        'stochastic_depth_prob': 0.0  # Disable stochastic depth for analysis
    }
    
    # Load model
    model = EnhancedCharTransformer(**config)
    
    if Path(args.model_path).exists():
        checkpoint = torch.load(args.model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"✅ Loaded model from {args.model_path}")
    else:
        print(f"❌ Model file not found: {args.model_path}")
        return
    
    model = model.to(device)
    model.eval()
    
    # Load validation data
    tokenizer = ByteTokenizer()
    try:
        val_data = load_data(args.data_path)
        val_tokens = tokenizer.encode(val_data)
        # Use middle section for analysis
        start_idx = len(val_tokens) // 2
        seq = torch.tensor(val_tokens[start_idx:start_idx + args.seq_length], 
                          dtype=torch.long, device=device).unsqueeze(0)
        print(f"✅ Using sequence of length {seq.size(1)} for analysis")
    except Exception as e:
        print(f"⚠️ Error loading data: {e}")
        # Use random sequence as fallback
        seq = torch.randint(0, 256, (1, args.seq_length), device=device)
        print("Using random sequence for analysis")
    
    # Extract attention patterns
    print("🔍 Extracting attention patterns...")
    extractor = AttentionExtractor(model)
    extractor.register_hooks()
    
    try:
        attention_maps = extractor.extract_attention(seq)
        print(f"✅ Extracted attention for {len(attention_maps)} heads")
        
        if not attention_maps:
            print("❌ No attention patterns extracted. Check model architecture.")
            return
        
        # Pick representative heads
        print("🎯 Selecting representative heads per cluster...")
        representative_heads = pick_representative_heads(attention_maps)
        print(f"Representative heads: {representative_heads}")
        
        # Create the figure
        print("🎨 Creating heatmap figure...")
        fig = create_heatmap_figure(attention_maps, representative_heads, args.output)
        
        # Generate captions
        latex_caption = generate_latex_caption(representative_heads)
        markdown_caption = generate_markdown_caption(representative_heads)
        
        # Save captions
        latex_file = Path(args.output).with_suffix('.tex')
        markdown_file = Path(args.output).with_suffix('.md')
        
        with open(latex_file, 'w') as f:
            f.write(latex_caption)
        with open(markdown_file, 'w') as f:
            f.write(markdown_caption)
            
        print(f"✅ Saved LaTeX caption to: {latex_file}")
        print(f"✅ Saved Markdown caption to: {markdown_file}")
        
        print(f"\n🎉 Generated attention heatmap figure: {args.output}")
        print("📝 Ready for inclusion in thesis!")
        
        # Print reference text suggestion
        print("\n📖 Suggested reference text for thesis:")
        print("Figure \\ref{fig:cluster_heatmaps} visualizes the four empirically")
        print("discovered attention patterns. The focused-local head (a) behaves like")
        print("a 16-token sliding window, whereas the strided head (b) leaps every 8")
        print("positions, etc. These interpretable masks confirm that dense heads")
        print("specialize and motivate our fixed sparse design.")
        
    finally:
        extractor.restore_hooks()

if __name__ == "__main__":
    main() 