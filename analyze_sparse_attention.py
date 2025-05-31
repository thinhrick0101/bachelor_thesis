import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sparse_attention import SparseTransformer
from transformer_imdb import TransformerModel
import imdb_loader
import os

def load_models(vocab_size, device='cuda'):
    """
    Load both sparse and dense transformer models.
    """
    # Sparse model config
    sparse_config = {
        'vocab_size': vocab_size,
        'embedding_dim': 512,
        'num_classes': 2,
        'num_heads': 8,
        'num_layers': 6,
        'ffn_dim': 2048,
        'dropout': 0.1,
        'attention_dropout': 0.1
    }
    
    # Dense model config
    dense_config = {
        'vocab_size': vocab_size,
        'd_model': 512,
        'nhead': 8,
        'num_layers': 6,
        'dim_feedforward': 2048,
        'num_classes': 2,
        'dropout': 0.1
    }
    
    # Initialize models
    sparse_model = SparseTransformer(**sparse_config).to(device)
    dense_model = TransformerModel(**dense_config).to(device)
    
    # Load trained weights
    sparse_model.load_state_dict(torch.load('models/sparse_transformer/best_model.pt'))
    dense_model.load_state_dict(torch.load('models/dense_transformer/best_model.pt'))
    
    return sparse_model, dense_model

def get_attention_patterns(model, input_text, tokenizer, device='cuda'):
    """
    Get attention patterns for a given input text.
    """
    model.eval()
    
    # Tokenize input
    tokens = tokenizer.encode(input_text)
    input_ids = torch.tensor(tokens).unsqueeze(0).to(device)
    
    # Get attention weights
    with torch.no_grad():
        if isinstance(model, SparseTransformer):
            _, attention_weights = model(input_ids, return_attention=True)
        else:
            # For dense model
            _, attention_weights = model(input_ids)
    
    return attention_weights

def visualize_attention_comparison(sparse_attn, dense_attn, tokens, layer_idx=0, head_idx=0, save_path=None):
    """
    Visualize and compare attention patterns between sparse and dense models.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot sparse attention
    sns.heatmap(sparse_attn[layer_idx][head_idx].cpu().numpy(),
                ax=ax1, cmap='viridis', xticklabels=tokens, yticklabels=tokens)
    ax1.set_title(f'Sparse Attention (Layer {layer_idx}, Head {head_idx})')
    
    # Plot dense attention
    sns.heatmap(dense_attn[layer_idx][head_idx].cpu().numpy(),
                ax=ax2, cmap='viridis', xticklabels=tokens, yticklabels=tokens)
    ax2.set_title(f'Dense Attention (Layer {layer_idx}, Head {head_idx})')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def analyze_sparsity(attention_weights):
    """
    Analyze sparsity patterns in attention weights.
    Returns:
    - Average sparsity (percentage of zero/near-zero weights)
    - Entropy of attention distributions
    """
    # Consider weights < 1e-6 as effectively zero
    threshold = 1e-6
    
    sparsity = []
    entropy = []
    
    for layer_weights in attention_weights:
        layer_sparsity = []
        layer_entropy = []
        
        for head_weights in layer_weights:
            # Calculate sparsity
            zeros = (head_weights < threshold).float().mean().item()
            layer_sparsity.append(zeros * 100)  # Convert to percentage
            
            # Calculate entropy
            # Normalize weights to get probability distribution
            probs = head_weights / head_weights.sum(dim=-1, keepdim=True)
            # Add small epsilon to avoid log(0)
            eps = 1e-10
            entropy_val = -(probs * torch.log(probs + eps)).sum(dim=-1).mean().item()
            layer_entropy.append(entropy_val)
        
        sparsity.append(layer_sparsity)
        entropy.append(layer_entropy)
    
    return np.array(sparsity), np.array(entropy)

def plot_sparsity_comparison(sparse_sparsity, dense_sparsity, save_path=None):
    """
    Plot sparsity comparison between sparse and dense models.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot average sparsity per layer
    layers = range(len(sparse_sparsity))
    ax1.plot(layers, sparse_sparsity.mean(axis=1), 'b-', label='Sparse Model')
    ax1.plot(layers, dense_sparsity.mean(axis=1), 'r-', label='Dense Model')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Average Sparsity (%)')
    ax1.set_title('Sparsity Comparison Across Layers')
    ax1.legend()
    ax1.grid(True)
    
    # Plot sparsity distribution
    ax2.boxplot([sparse_sparsity.flatten(), dense_sparsity.flatten()],
                labels=['Sparse Model', 'Dense Model'])
    ax2.set_ylabel('Sparsity (%)')
    ax2.set_title('Sparsity Distribution')
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_entropy_comparison(sparse_entropy, dense_entropy, save_path=None):
    """
    Plot entropy comparison between sparse and dense models.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot average entropy per layer
    layers = range(len(sparse_entropy))
    ax1.plot(layers, sparse_entropy.mean(axis=1), 'b-', label='Sparse Model')
    ax1.plot(layers, dense_entropy.mean(axis=1), 'r-', label='Dense Model')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Average Entropy')
    ax1.set_title('Entropy Comparison Across Layers')
    ax1.legend()
    ax1.grid(True)
    
    # Plot entropy distribution
    ax2.boxplot([sparse_entropy.flatten(), dense_entropy.flatten()],
                labels=['Sparse Model', 'Dense Model'])
    ax2.set_ylabel('Entropy')
    ax2.set_title('Entropy Distribution')
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def main():
    # Create output directory
    os.makedirs('attention_analysis/sparse_comparison', exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data and models
    train_data, val_data, test_data, vocab_size = imdb_loader.load_imdb_data()
    tokenizer = imdb_loader.get_tokenizer()
    sparse_model, dense_model = load_models(vocab_size, device)
    
    # Example text for visualization
    example_texts = [
        "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout.",
        "I couldn't stand this film. The dialogue was terrible and the pacing was all wrong.",
        "While the movie had some good moments, overall it was just average. Nothing special but not terrible either."
    ]
    
    # Analyze attention patterns for each example
    for idx, text in enumerate(example_texts):
        # Get attention patterns
        sparse_attention = get_attention_patterns(sparse_model, text, tokenizer, device)
        dense_attention = get_attention_patterns(dense_model, text, tokenizer, device)
        
        # Get tokens for visualization
        tokens = tokenizer.encode(text)
        token_strings = [tokenizer.decode([t]) for t in tokens]
        
        # Visualize attention patterns
        for layer in range(min(3, len(sparse_attention))):  # Show first 3 layers
            for head in range(min(4, sparse_attention[layer].size(1))):  # Show first 4 heads
                save_path = f'attention_analysis/sparse_comparison/attention_example{idx}_layer{layer}_head{head}.png'
                visualize_attention_comparison(
                    sparse_attention,
                    dense_attention,
                    token_strings,
                    layer_idx=layer,
                    head_idx=head,
                    save_path=save_path
                )
        
        # Analyze sparsity and entropy
        sparse_sparsity, sparse_entropy = analyze_sparsity(sparse_attention)
        dense_sparsity, dense_entropy = analyze_sparsity(dense_attention)
        
        # Plot comparisons
        plot_sparsity_comparison(
            sparse_sparsity,
            dense_sparsity,
            save_path=f'attention_analysis/sparse_comparison/sparsity_example{idx}.png'
        )
        
        plot_entropy_comparison(
            sparse_entropy,
            dense_entropy,
            save_path=f'attention_analysis/sparse_comparison/entropy_example{idx}.png'
        )
        
        # Print statistics
        print(f"\nExample {idx + 1}:")
        print(f"Sparse Model - Average Sparsity: {sparse_sparsity.mean():.2f}%, Average Entropy: {sparse_entropy.mean():.2f}")
        print(f"Dense Model - Average Sparsity: {dense_sparsity.mean():.2f}%, Average Entropy: {dense_entropy.mean():.2f}")

if __name__ == '__main__':
    main() 