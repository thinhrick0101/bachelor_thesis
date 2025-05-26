import json
import numpy as np
from pathlib import Path

def save_attention_metrics(sparsity_stats, output_file='attention_metrics.json'):
    """
    Save attention metrics in a structured JSON format
    """
    metrics = {
        'layers': {},
        'summary': {
            'average_attention': {},
            'entropy': {},
            'sparsity': {},
            'max_attention': {}
        }
    }
    
    # Process each layer
    for layer_idx, layer_stats in enumerate(sparsity_stats):
        layer_metrics = {
            'heads': {},
            'layer_summary': {
                'avg_attention': np.mean([h['avg_attention'] for h in layer_stats]),
                'avg_entropy': np.mean([h['entropy'] for h in layer_stats]),
                'avg_sparsity': np.mean([h['sparsity_90'] for h in layer_stats]),
                'avg_max_attention': np.mean([h['max_attention'] for h in layer_stats])
            }
        }
        
        # Process each head in the layer
        for head_stats in layer_stats:
            head_idx = head_stats['head']
            layer_metrics['heads'][f'head_{head_idx}'] = {
                'average_attention': head_stats['avg_attention'],
                'entropy': head_stats['entropy'],
                'sparsity_90': head_stats['sparsity_90'],
                'max_attention': head_stats['max_attention']
            }
        
        metrics['layers'][f'layer_{layer_idx}'] = layer_metrics
    
    # Calculate global statistics
    all_heads = [head for layer in sparsity_stats for head in layer]
    metrics['summary'] = {
        'global_stats': {
            'mean_attention': np.mean([h['avg_attention'] for h in all_heads]),
            'mean_entropy': np.mean([h['entropy'] for h in all_heads]),
            'mean_sparsity': np.mean([h['sparsity_90'] for h in all_heads]),
            'mean_max_attention': np.mean([h['max_attention'] for h in all_heads]),
            'std_attention': np.std([h['avg_attention'] for h in all_heads]),
            'std_entropy': np.std([h['entropy'] for h in all_heads]),
            'std_sparsity': np.std([h['sparsity_90'] for h in all_heads]),
            'std_max_attention': np.std([h['max_attention'] for h in all_heads])
        },
        'extremes': {
            'max_entropy': max(h['entropy'] for h in all_heads),
            'min_entropy': min(h['entropy'] for h in all_heads),
            'max_sparsity': max(h['sparsity_90'] for h in all_heads),
            'min_sparsity': min(h['sparsity_90'] for h in all_heads)
        }
    }
    
    # Save to file
    output_path = Path('attention_analysis') / output_file
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Also save a human-readable summary
    summary_path = output_path.with_suffix('.txt')
    with open(summary_path, 'w') as f:
        f.write("=== Attention Pattern Analysis Summary ===\n\n")
        
        # Global statistics
        f.write("Global Statistics:\n")
        f.write("-----------------\n")
        for stat, value in metrics['summary']['global_stats'].items():
            f.write(f"{stat}: {value:.4f}\n")
        f.write("\n")
        
        # Extreme values
        f.write("Extreme Values:\n")
        f.write("--------------\n")
        for stat, value in metrics['summary']['extremes'].items():
            f.write(f"{stat}: {value:.4f}\n")
        f.write("\n")
        
        # Layer-wise summary
        f.write("Layer-wise Summary:\n")
        f.write("-----------------\n")
        for layer_name, layer_data in metrics['layers'].items():
            f.write(f"\n{layer_name}:\n")
            for metric, value in layer_data['layer_summary'].items():
                f.write(f"  {metric}: {value:.4f}\n")
            
            # Head statistics
            f.write("\n  Head Statistics:\n")
            for head_name, head_data in layer_data['heads'].items():
                f.write(f"  {head_name}:\n")
                for metric, value in head_data.items():
                    f.write(f"    {metric}: {value:.4f}\n")
                f.write("\n")

def main():
    """
    Main function to load attention patterns and save metrics
    """
    from attention_analysis import analyze_attention_sparsity, get_attention_patterns, load_model
    from stable_char_transformer import ByteTokenizer
    import torch
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'bachelor_thesis/models/dense_byte_transformer.pt'
    
    # Load model and tokenizer
    print("Loading model...")
    model = load_model(model_path, device)
    tokenizer = ByteTokenizer()
    
    # Sample text for analysis
    text = """The quick brown fox jumps over the lazy dog. This pangram contains every letter of the English alphabet at least once. Pangrams are often used to display font samples and test keyboards."""
    
    # Get attention patterns
    print("Extracting attention patterns...")
    attention_patterns = get_attention_patterns(model, text, tokenizer, device)
    
    # Analyze patterns
    print("Analyzing attention patterns...")
    sparsity_stats = analyze_attention_sparsity(attention_patterns)
    
    # Save metrics
    print("Saving attention metrics...")
    save_attention_metrics(sparsity_stats)
    
    print("Analysis complete! Check 'attention_analysis/attention_metrics.json' and 'attention_analysis/attention_metrics.txt' for results.")

if __name__ == "__main__":
    main() 