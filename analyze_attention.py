import torch
import torch.nn as nn
import os
from stable_char_transformer import (
    EnhancedCharTransformer, 
    ByteTokenizer, 
    create_batches, 
    load_data,
    train_model
)
from attention_analysis import (
    AttentionPatternAnalyzer,
    CustomSparseAttention,
    visualize_attention_patterns,
    compare_attention_mechanisms,
    plot_comparison_results
)

def main():
    # Setup paths
    print("Setting up paths...")
    model_dir = 'bachelor_thesis'
    model_path = os.path.join(model_dir, 'enhanced_char_transformer_model.pt')
    os.makedirs(model_dir, exist_ok=True)
    
    # Model configuration (matching the pretrained model)
    config = {
        'vocab_size': 256,  # Keep at 256 for byte-level tokenization
        'd_model': 512,     # Keep at 512 for our analysis
        'nhead': 8,
        'num_layers': 12,   # Keep at 12 for our analysis
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'attention_dropout': 0.1,
        'activation_dropout': 0.1,
        'token_dropout': 0.05,
        'use_checkpoint': True,
        'stochastic_depth_prob': 0.1
    }
    
    # Create model instance
    model = EnhancedCharTransformer(**config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create tokenizer
    tokenizer = ByteTokenizer()
    
    # Check if we need to train the model
    if not os.path.exists(model_path):
        print("No pretrained model found. Training new model...")
        
        # Load training data
        print("Loading training data...")
        train_text = load_data('bachelor_thesis/data/enwik8')
        
        # Split into train/val
        split_idx = int(len(train_text) * 0.9)
        train_data = tokenizer.encode(train_text[:split_idx])
        val_data = tokenizer.encode(train_text[split_idx:])
        
        # Create batches
        batch_size = 32
        seq_length = 1024
        train_batches = create_batches(train_data, batch_size, seq_length)
        val_batches = create_batches(val_data, batch_size, seq_length)
        
        # Train model
        print("Training model...")
        model, _ = train_model(
            model=model,
            train_batches=train_batches,
            val_batches=val_batches,
            num_epochs=5,  # Reduced for faster training
            learning_rate=1e-4,
            weight_decay=0.1,
            warmup_steps=1000,
            device=device
        )
        
        # Save model
        print(f"Saving model to {model_path}")
        torch.save(model.state_dict(), model_path)
    else:
        print("Loading pretrained model...")
        try:
            model.load_state_dict(torch.load(model_path))
        except Exception as e:
            print(f"Failed to load pretrained model: {e}")
            print("Training new model instead...")
            
            # Load training data
            print("Loading training data...")
            train_text = load_data('bachelor_thesis/data/enwik8')
            
            # Split into train/val
            split_idx = int(len(train_text) * 0.9)
            train_data = tokenizer.encode(train_text[:split_idx])
            val_data = tokenizer.encode(train_text[split_idx:])
            
            # Create batches
            batch_size = 32
            seq_length = 1024
            train_batches = create_batches(train_data, batch_size, seq_length)
            val_batches = create_batches(val_data, batch_size, seq_length)
            
            # Train model
            print("Training model...")
            model, _ = train_model(
                model=model,
                train_batches=train_batches,
                val_batches=val_batches,
                num_epochs=5,  # Reduced for faster training
                learning_rate=1e-4,
                weight_decay=0.1,
                warmup_steps=1000,
                device=device
            )
            
            # Save model
            print(f"Saving model to {model_path}")
            torch.save(model.state_dict(), model_path)
    
    model = model.to(device)
    
    # Load test data
    print("Loading test data...")
    test_text = load_data('bachelor_thesis/data/enwik8')[-10000:]  # Use last 10k bytes for testing
    test_data = tokenizer.encode(test_text)
    
    # Create test batches
    batch_size = 32
    seq_length = 1024
    test_batches = create_batches(test_data, batch_size, seq_length)
    
    # Analyze attention patterns
    print("\nAnalyzing attention patterns...")
    analyzer = AttentionPatternAnalyzer(model)
    
    # Prepare a sample sequence for analysis
    sample_seq = test_batches[0][0].to(device)  # Use first batch input
    results = analyzer.analyze_sequence(sample_seq, tokenizer)
    
    # Visualize the analysis results
    print("Visualizing attention patterns...")
    visualize_attention_patterns(results, os.path.join(model_dir, 'attention_analysis.png'))
    
    # Create sparse attention model based on analysis
    print("\nCreating sparse attention model...")
    
    # Use the analysis results to configure sparse attention
    local_patterns = results['local_patterns']
    sparsity_potential = results['sparsity_potential']
    
    # Calculate median effective window size across layers
    window_sizes = [layer_data['median_window'] for layer_data in local_patterns.values()]
    median_window = int(sum(window_sizes) / len(window_sizes))
    
    # Calculate average number of global tokens needed
    global_tokens = [layer_data['top_k_90'] for layer_data in sparsity_potential.values()]
    avg_global_tokens = int(sum(global_tokens) / len(global_tokens))
    
    # Create sparse attention configuration
    sparse_config = config.copy()
    sparse_config['attention_class'] = CustomSparseAttention
    sparse_config['attention_kwargs'] = {
        'local_window_size': median_window,
        'num_global_tokens': avg_global_tokens
    }
    
    # Create sparse model
    sparse_model = EnhancedCharTransformer(**sparse_config)
    
    # Initialize sparse model with pretrained weights where possible
    pretrained_dict = model.state_dict()
    sparse_dict = sparse_model.state_dict()
    
    # Filter out attention parameters that don't match
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in sparse_dict and v.shape == sparse_dict[k].shape}
    sparse_dict.update(pretrained_dict)
    sparse_model.load_state_dict(sparse_dict)
    
    sparse_model = sparse_model.to(device)
    
    # Compare performance
    print("\nComparing attention mechanisms...")
    comparison_results = compare_attention_mechanisms(
        model, sparse_model, test_batches, tokenizer
    )
    
    # Plot comparison results
    plot_comparison_results(comparison_results, os.path.join(model_dir, 'attention_comparison.png'))
    
    # Print numerical results
    print("\nResults Summary:")
    print("Full Attention:")
    print(f"- Perplexity: {comparison_results['perplexity']['full'][0]:.2f}")
    print(f"- Speed: {comparison_results['speed']['full'][0]:.2f} tokens/sec")
    print(f"- Memory: {comparison_results['memory']['full'][0]:.2f} MB")
    
    print("\nSparse Attention:")
    print(f"- Perplexity: {comparison_results['perplexity']['sparse'][0]:.2f}")
    print(f"- Speed: {comparison_results['speed']['sparse'][0]:.2f} tokens/sec")
    print(f"- Memory: {comparison_results['memory']['sparse'][0]:.2f} MB")
    
    # Calculate relative differences
    perplexity_diff = (comparison_results['perplexity']['sparse'][0] / comparison_results['perplexity']['full'][0] - 1) * 100
    speed_improvement = (comparison_results['speed']['sparse'][0] / comparison_results['speed']['full'][0] - 1) * 100
    memory_reduction = (1 - comparison_results['memory']['sparse'][0] / comparison_results['memory']['full'][0]) * 100
    
    print("\nRelative Differences:")
    print(f"- Perplexity degradation: {perplexity_diff:.1f}%")
    print(f"- Speed improvement: {speed_improvement:.1f}%")
    print(f"- Memory reduction: {memory_reduction:.1f}%")

if __name__ == "__main__":
    main() 