import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import math
import os
import argparse
from torch.cuda.amp import autocast, GradScaler
from stable_char_transformer import (
    EnhancedCharTransformer,
    create_batches,
    train_model,
    visualize_results,
    load_data
)
from bpe_tokenizer import BPETokenizer

def main():
    parser = argparse.ArgumentParser(description='Train Enhanced Transformer Model')
    
    # Data parameters
    parser.add_argument('--data', type=str, default='data/enwik8',
                      help='Path to training data')
    parser.add_argument('--tokenizer', type=str, required=True,
                      help='Path to pre-trained tokenizer')
    parser.add_argument('--max-chars', type=int, default=20000000,
                      help='Maximum number of characters to use for training')
    
    # Model parameters
    parser.add_argument('--d-model', type=int, default=768,
                      help='Model dimension')
    parser.add_argument('--nhead', type=int, default=12,
                      help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=16,
                      help='Number of transformer layers')
    parser.add_argument('--dim-feedforward', type=int, default=3072,
                      help='Dimension of feedforward network')
    parser.add_argument('--dropout', type=float, default=0.2,
                      help='Dropout rate')
    parser.add_argument('--attention-dropout', type=float, default=0.15,
                      help='Attention dropout rate')
    parser.add_argument('--activation-dropout', type=float, default=0.15,
                      help='Activation dropout rate')
    parser.add_argument('--token-dropout', type=float, default=0.1,
                      help='Token dropout rate')
    parser.add_argument('--stochastic-depth-prob', type=float, default=0.1,
                      help='Stochastic depth probability')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size')
    parser.add_argument('--seq-length', type=int, default=512,
                      help='Sequence length')
    parser.add_argument('--num-epochs', type=int, default=100,
                      help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=5e-4,
                      help='Learning rate')
    parser.add_argument('--min-lr', type=float, default=1e-5,
                      help='Minimum learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.1,
                      help='Weight decay')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                      help='Label smoothing factor')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=8,
                      help='Number of gradient accumulation steps')
    parser.add_argument('--warmup-epochs', type=int, default=2,
                      help='Number of warmup epochs')
    parser.add_argument('--output-dir', type=str, default='models',
                      help='Output directory for model and visualizations')
    parser.add_argument('--no-mixed-precision', action='store_true',
                      help='Disable mixed precision training')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and prepare data
    print("Loading data...")
    text = load_data(args.data)
    if args.max_chars and len(text) > args.max_chars:
        print(f"Limiting data to first {args.max_chars:,} characters")
        text = text[:args.max_chars]
    
    # Load pre-trained tokenizer
    print(f"Loading tokenizer from {args.tokenizer}")
    tokenizer = BPETokenizer()
    tokenizer.load(args.tokenizer)
    vocab_size = len(tokenizer.encoder)
    print(f"Loaded tokenizer with vocabulary size: {vocab_size:,}")
    
    # Encode the text
    print("Encoding text...")
    data = tokenizer.encode(text)
    
    # Split data into training and validation sets (90% / 10%)
    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    # Create batches
    print("Creating batches...")
    train_batches = create_batches(train_data, args.batch_size, args.seq_length)
    val_batches = create_batches(val_data, args.batch_size, args.seq_length)
    print(f"Created {len(train_batches)} training batches and {len(val_batches)} validation batches")
    
    # Calculate warmup steps
    warmup_steps = len(train_batches) * args.warmup_epochs
    
    # Initialize model
    print("Initializing model...")
    model = EnhancedCharTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        activation_dropout=args.activation_dropout,
        token_dropout=args.token_dropout,
        use_checkpoint=True,
        stochastic_depth_prob=args.stochastic_depth_prob
    )
    
    # Print model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {trainable_params:,} trainable out of {total_params:,} total")
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Train model
    print("\n=== Training Enhanced Transformer Model ===")
    model, (train_losses, val_losses) = train_model(
        model=model,
        train_batches=train_batches,
        val_batches=val_batches,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        min_lr=args.min_lr,
        device=device,
        patience=8,
        label_smoothing=args.label_smoothing,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_mixed_precision=not args.no_mixed_precision,
        use_cosine_schedule=True
    )
    
    # Save visualization
    vis_path = os.path.join(args.output_dir, 'training_loss.png')
    visualize_results(train_losses, val_losses, vis_path)
    print(f"\nTraining visualization saved to {vis_path}")
    
    # Save model
    model_path = os.path.join(args.output_dir, 'transformer_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'd_model': args.d_model,
        'nhead': args.nhead,
        'num_layers': args.num_layers,
        'dim_feedforward': args.dim_feedforward,
        'dropout': args.dropout,
        'attention_dropout': args.attention_dropout,
        'activation_dropout': args.activation_dropout,
        'token_dropout': args.token_dropout,
        'stochastic_depth_prob': args.stochastic_depth_prob
    }, model_path)
    print(f"Model saved to {model_path}")
    
    # Test generation
    print("\n=== Testing Text Generation ===")
    prompts = [
        "The quick brown fox",
        "In the beginning",
        "Once upon a time"
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        for temp in [0.7, 0.9]:
            generated_text = model.generate(
                prompt=prompt,
                max_length=200,
                temperature=temp,
                top_k=50,
                top_p=0.92,
                repetition_penalty=1.2,
                tokenizer=tokenizer,
                device=device
            )
            print(f"\nTemperature {temp}:")
            print(generated_text)

if __name__ == '__main__':
    main() 