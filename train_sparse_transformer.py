import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import math
import argparse
from stable_char_transformer import (
    SparseTransformer, 
    ByteTokenizer, 
    create_batches, 
    load_data,
    train_model
)
from sparse_byte_transformer import SparseByteTransformer
from torch.cuda.amp import autocast

def visualize_loss(train_losses, val_losses=None, output_file='sparse_model_loss.png'):
    """Visualize training and validation losses"""
    plt.figure(figsize=(12, 6))
    
    # Plot training loss
    plt.plot(train_losses, label='Training Loss', marker='o', markersize=4, linestyle='-', linewidth=1)
    
    # Plot validation loss if available
    if val_losses:
        plt.plot(val_losses, label='Validation Loss', marker='s', markersize=4, linestyle='-', linewidth=1)
        
        # Plot best validation loss point
        best_epoch = val_losses.index(min(val_losses))
        best_loss = val_losses[best_epoch]
        plt.plot(best_epoch, best_loss, 'r*', markersize=10, label=f'Best Val Loss: {best_loss:.4f}')
    
    plt.title('Model Training History', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add perplexity as secondary y-axis
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # Create perplexity ticks based on loss values
    loss_ticks = ax1.get_yticks()
    perplexity_ticks = [math.exp(x) for x in loss_ticks if x > 0]
    ax2.set_yticks(perplexity_ticks)
    ax2.set_yticklabels([f'{x:.1f}' for x in perplexity_ticks])
    ax2.set_ylabel('Perplexity', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print statistics
    print("\nTraining Statistics:")
    print(f"Initial Loss: {train_losses[0]:.4f}")
    print(f"Final Loss: {train_losses[-1]:.4f}")
    print(f"Best Loss: {min(train_losses):.4f}")
    
    if val_losses:
        print("\nValidation Statistics:")
        print(f"Initial Loss: {val_losses[0]:.4f}")
        print(f"Final Loss: {val_losses[-1]:.4f}")
        print(f"Best Loss: {min(val_losses):.4f}")

def generate_text(model, tokenizer, prompt, max_length=1000, temperature=0.7, top_k=50, top_p=0.9, device='cuda'):
    """Generate text using the trained sparse transformer with improved sampling"""
    model.eval()
    
    # Encode the prompt
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    generated = input_tensor
    
    # Track recent tokens for repetition detection
    recent_tokens = []
    token_counts = {}
    
    # Generate text token by token
    with torch.no_grad():
        for _ in range(max_length):
            # Get model predictions
            with autocast():
                logits = model(generated)
                next_token_logits = logits[0, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Dynamic repetition penalty based on recent usage
            for token in set(recent_tokens):
                count = recent_tokens.count(token)
                penalty = 1.0 + (count * 0.5)  # Increased penalty for frequency
                next_token_logits[token] /= penalty
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = next_token.item()
            
            # Update tracking
            recent_tokens.append(token_id)
            if len(recent_tokens) > 20:  # Track last 20 tokens
                recent_tokens.pop(0)
            
            token_counts[token_id] = token_counts.get(token_id, 0) + 1
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            
            # Stop conditions
            if token_id in [10, 0]:  # newline or end token
                break
                
            # Check for repetitive patterns
            if len(recent_tokens) >= 5:
                # Check for immediate repetition
                if len(set(recent_tokens[-5:])) == 1:
                    break
                    
                # Check for bi-gram repetition
                if len(recent_tokens) >= 10:
                    last_bigrams = [tuple(recent_tokens[i:i+2]) for i in range(len(recent_tokens)-2)]
                    if len(set(last_bigrams)) <= 2:
                        break
            
            # Check for overuse of any token
            max_count = max(token_counts.values()) if token_counts else 0
            if max_count > len(generated[0]) * 0.3:  # No token should be >30% of generation
                break
    
    # Decode and return the generated text
    generated_ids = generated[0].tolist()
    return tokenizer.decode(generated_ids)

def main():
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'

    # Model configuration
    config_dict = {
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
        'stochastic_depth_prob': 0.1,
        'seq_length': 1024  # Added: SparseByteTransformer needs this for PositionalEncoding
    }
    
    # Convert dict to Namespace for attribute access in the model
    model_config = argparse.Namespace(**config_dict)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    #model = SparseTransformer(**config)
    # Create model instance
    model = SparseByteTransformer(model_config)
    model = model.to(device)
    
    # Create tokenizer
    tokenizer = ByteTokenizer()
    
    # Check if model exists
    model_path = 'bachelor_thesis/models/sparse_byte_transformer.pt'
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        checkpoint = torch.load(model_path)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded model checkpoint with training history")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded model weights only")
    else:
        # Load training data
        print("Loading training data...")
        train_text = load_data('data/enwik8')
        
        # Split into train/val
        split_idx = int(len(train_text) * 0.9)
        train_data = tokenizer.encode(train_text[:split_idx])
        val_data = tokenizer.encode(train_text[split_idx:])
        
        # Create batches
        batch_size = 16
        seq_length = 512
        train_batches = create_batches(train_data, batch_size, seq_length)
        val_batches = create_batches(val_data, batch_size, seq_length)
        
        # Train model
        print("Training model...")
        model, (train_losses, val_losses) = train_model(
            model=model,
            train_batches=train_batches,
            val_batches=val_batches,
            num_epochs=40,  # Full training run
            learning_rate=1e-4,
            weight_decay=0.1,
            warmup_steps=1000,
            device=device,
            patience=5,  # Increased patience for longer training
            min_lr=1e-5,  # Minimum learning rate
            gradient_accumulation_steps=4,  # Gradient accumulation for stability
            use_mixed_precision=True,  # Use mixed precision training
            use_cosine_schedule=True  # Use cosine learning rate schedule
        )
        
        # Save model and loss history
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        print(f"Saving model to {model_path}")
        torch.save(model.state_dict(), model_path)
        
        # Save training history separately
        history_path = 'bachelor_thesis/models/sparse_byte_transformer_history.pt'
        torch.save({
            'train_losses': train_losses,
            'val_losses': val_losses
        }, history_path)
        
        # Visualize training history
        print("Generating loss plot...")
        visualize_loss(train_losses, val_losses, 'sparse_model_training_loss.png')
    
    # Generate example texts with different temperatures
    print("\nGenerating example texts with different temperatures:")
    example_prompts = [
        "The movie was",
        "In the beginning",
        "She looked at",
        "The system could",
        "Deep learning is"
    ]
    
    temperatures = [0.6, 0.8, 1.0]
    max_length = 200
    
    print("\nGenerating samples with different temperatures:")
    for prompt in example_prompts:
        print(f"\nPrompt: {prompt}")
        for temp in temperatures:
            print(f"\nTemperature {temp}:")
            try:
                generated = generate_text(
                    model, 
                    tokenizer, 
                    prompt, 
                    temperature=temp,
                    max_length=max_length
                )
                print(generated)
            except Exception as e:
                print(f"Error generating text: {str(e)}")
    
    # Check if we're in an interactive environment
    import sys
    if sys.stdin.isatty():
        print("\nEntering interactive mode (Ctrl+C to exit)")
        try:
            while True:
                try:
                    prompt = input("\nPrompt: ")
                    temp = float(input("Temperature (0.1-1.0): "))
                    length = int(input("Maximum length: "))
                    
                    generated = generate_text(
                        model, 
                        tokenizer, 
                        prompt, 
                        temperature=temp,
                        max_length=length
                    )
                    print("\nGenerated text:")
                    print(generated)
                except (KeyboardInterrupt, EOFError):
                    print("\nExiting interactive mode...")
                    break
                except ValueError as e:
                    print(f"Invalid input: {str(e)}")
                except Exception as e:
                    print(f"Error: {str(e)}")
        except KeyboardInterrupt:
            print("\nExiting...")
    else:
        print("\nRunning in non-interactive mode, skipping interactive prompt")

if __name__ == "__main__":
    main()