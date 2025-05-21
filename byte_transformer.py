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
def main():
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

        
        # Load training data
    print("Loading training data...")
    train_text = load_data('data/enwik8')
        
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