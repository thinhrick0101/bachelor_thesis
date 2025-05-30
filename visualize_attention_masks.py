import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sparse_transformer_v2 import SparseMultiheadAttention

def plot_attention_mask(mask: torch.Tensor, title: str, save_path: Path):
    """Plot an attention mask as a heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(mask.cpu().numpy(), cmap='viridis', cbar=True)
    plt.title(title)
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.savefig(save_path)
    plt.close()

def visualize_attention_patterns(seq_length: int = 64):
    """Visualize different attention patterns."""
    
    # Create output directory
    output_dir = Path('attention_visualization')
    output_dir.mkdir(exist_ok=True)
    
    # Initialize attention module
    attention = SparseMultiheadAttention(
        embed_dim=512,  # Arbitrary embedding dimension
        num_heads=8,
        max_seq_length=seq_length
    )
    
    # Visualize masks for each head
    head_descriptions = {
        0: "Local Attention (Head 0)",
        1: "Local Attention (Head 1)",
        2: "Local Attention (Head 2)",
        3: "Local Attention (Head 3)",
        4: "Strided Attention (Head 4)",
        5: "Strided Attention (Head 5)",
        6: "Global Attention (Head 6)",
        7: "Global Attention (Head 7)"
    }
    
    # Plot each attention pattern
    for head_idx, description in head_descriptions.items():
        # Get mask for this head
        mask = attention._get_mask_for_head(head_idx, seq_length)
        
        # Save visualization
        save_path = output_dir / f'head_{head_idx}_pattern.png'
        plot_attention_mask(mask, description, save_path)
        
        # Print statistics
        total_connections = mask.sum().item()
        sparsity = 1.0 - (total_connections / (seq_length * seq_length))
        avg_connections_per_token = total_connections / seq_length
        
        print(f"\n{description}:")
        print(f"Sparsity: {sparsity:.2%}")
        print(f"Average connections per token: {avg_connections_per_token:.1f}")
        
        # Analyze local vs global attention
        local_window = attention.local_window
        positions = torch.arange(seq_length)
        distances = (positions.unsqueeze(1) - positions.unsqueeze(0)).abs()
        local_mask = distances <= local_window
        
        local_attention = (mask & local_mask).sum().item()
        global_attention = (mask & ~local_mask).sum().item()
        
        print(f"Local attention ratio: {local_attention/total_connections:.2%}")
        print(f"Global attention ratio: {global_attention/total_connections:.2%}")

def main():
    print("Visualizing attention patterns...")
    print("-" * 50)
    
    # Visualize for different sequence lengths
    for seq_length in [32, 64, 128]:
        print(f"\nSequence length: {seq_length}")
        print("-" * 50)
        visualize_attention_patterns(seq_length)

if __name__ == '__main__':
    main() 