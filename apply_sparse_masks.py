import torch
from sparse_attention_masks import convert_to_sparse_transformer
from sparse_char_transformer import SparseCharTransformer

def main():
    # Load the trained model
    checkpoint = torch.load('models/best_sparse_transformer.pt')
    
    # Create model instance
    model_config = {
        'vocab_size': 256,
        'd_model': 512,
        'nhead': 8,
        'num_layers': 12,
        'dim_feedforward': 1024,
        'dropout': 0.1
    }
    
    model = SparseCharTransformer(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Convert to use sparse attention
    print("Converting model to use sparse attention patterns...")
    sparse_model = convert_to_sparse_transformer(model, max_seq_length=1024)
    
    # Save the converted model
    torch.save({
        'model_state_dict': sparse_model.state_dict(),
        'config': model_config,
        'sparse_config': {
            'max_seq_length': 1024,
            'local_window': 16,
            'stride': 8,
            'num_global_tokens': 4
        }
    }, 'models/sparse_transformer_with_masks.pt')
    
    print("Model converted and saved with sparse attention masks!")
    
    # Print summary of attention patterns
    print("\nAttention Pattern Summary:")
    print("- Heads 0-3: Local attention (window size: ±16)")
    print("- Heads 4-5: Strided attention (local + periodic every 8 tokens)")
    print("- Head 6: Global attention (4 anchor tokens)")
    print("- Head 7: Random sparse attention (80% sparsity)")
    
if __name__ == "__main__":
    main() 