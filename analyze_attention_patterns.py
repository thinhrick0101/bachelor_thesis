import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sparse_char_transformer import SparseCharTransformer
from sparse_attention_masks import StaticSparseMask

def plot_attention_pattern(mask, title, save_path):
    """Plot a single attention pattern."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(mask.cpu().numpy(), cmap='viridis', cbar=True)
    plt.title(title)
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.savefig(save_path)
    plt.close()

def analyze_attention_patterns(seq_length=128):
    """Analyze and visualize different attention patterns."""
    
    # Create output directory
    output_dir = Path('attention_analysis')
    output_dir.mkdir(exist_ok=True)
    
    # Initialize static mask generator
    mask_gen = StaticSparseMask(
        max_seq_length=seq_length,
        num_heads=8,
        local_window=16,
        stride=8,
        num_global_tokens=4,
        device='cpu'
    )
    
    # Analyze each head type
    head_types = {
        'local': (0, 'Local Attention (±16 window)'),
        'local2': (1, 'Local Attention with Different Seed'),
        'local3': (2, 'Local Attention Pattern 3'),
        'local4': (3, 'Local Attention Pattern 4'),
        'strided1': (4, 'Strided Attention 1'),
        'strided2': (5, 'Strided Attention 2'),
        'global': (6, 'Global Attention (4 anchors)'),
        'random': (7, 'Random Sparse Attention (80% sparse)')
    }
    
    # Plot each attention pattern
    for name, (head_idx, title) in head_types.items():
        mask = mask_gen.get_mask_for_head(head_idx, seq_length)
        save_path = output_dir / f'{name}_pattern.png'
        plot_attention_pattern(mask, title, save_path)
        
        # Calculate and print statistics
        sparsity = (~mask).float().mean().item()
        avg_connections = mask.float().sum(dim=1).mean().item()
        print(f"\nPattern: {title}")
        print(f"Sparsity: {sparsity:.2%}")
        print(f"Average connections per token: {avg_connections:.1f}")
        
        # Analyze local vs global attention
        local_window = 16
        positions = torch.arange(seq_length)
        distances = (positions.unsqueeze(1) - positions.unsqueeze(0)).abs()
        local_mask = distances <= local_window
        
        local_attention = (mask & local_mask).float().sum().item()
        global_attention = (mask & ~local_mask).float().sum().item()
        total_attention = mask.float().sum().item()
        
        print(f"Local attention ratio: {local_attention/total_attention:.2%}")
        print(f"Global attention ratio: {global_attention/total_attention:.2%}")

def analyze_model_attention(model_path='models/sparse_transformer_with_masks.pt', 
                          sample_text="The quick brown fox jumps over the lazy dog."):
    """Analyze attention patterns from a trained model."""
    
    # Load model
    checkpoint = torch.load(model_path)
    model_config = checkpoint['config']
    model = SparseCharTransformer(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Convert text to tensor
    input_ids = torch.tensor([[ord(c) for c in sample_text]], dtype=torch.long)
    
    # Create attention mask
    src_mask = torch.ones(1, len(sample_text), len(sample_text))
    
    # Get attention weights
    with torch.no_grad():
        output = model(input_ids, src_mask=src_mask)
        attention_weights = model.get_attention_weights()
    
    # Create output directory for model analysis
    output_dir = Path('attention_analysis/model')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Plot attention weights for each layer and head
    for layer_idx, layer_attn in enumerate(attention_weights):
        layer_dir = output_dir / f'layer_{layer_idx}'
        layer_dir.mkdir(exist_ok=True)
        
        for head_idx in range(layer_attn.size(1)):
            attn_map = layer_attn[0, head_idx]  # [seq_len, seq_len]
            save_path = layer_dir / f'head_{head_idx}_pattern.png'
            title = f'Layer {layer_idx}, Head {head_idx} Attention Pattern'
            plot_attention_pattern(attn_map, title, save_path)
            
            # Calculate statistics
            sparsity = (attn_map < 0.01).float().mean().item()
            print(f"\n{title}")
            print(f"Effective sparsity: {sparsity:.2%}")
            print(f"Max attention weight: {attn_map.max().item():.3f}")
            print(f"Mean attention weight: {attn_map.mean().item():.3f}")

def main():
    print("Analyzing static attention patterns...")
    analyze_attention_patterns()
    
    print("\nAnalyzing model attention patterns...")
    try:
        analyze_model_attention()
    except FileNotFoundError:
        print("No trained model found. Please train the model first.")

if __name__ == '__main__':
    main() 