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

def generate_text(model, tokenizer, prompt, max_length=1000, temperature=0.7, top_k=50, top_p=0.9, device='cuda'):
    """Generate text using the trained model"""
    model.eval()
    
    # Encode the prompt
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    
    # Generate text
    with torch.no_grad():
        output = model.generate(
            prompt=input_tensor,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            tokenizer=tokenizer,
            device=device
        )
    
    # Decode and return the generated text
    return tokenizer.decode(output[0].tolist())

def main():
    # Model configuration
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
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model instance
    model = EnhancedCharTransformer(**config)
    model = model.to(device)
    
    # Create tokenizer
    tokenizer = ByteTokenizer()
    
    # Check if model exists
    model_path = 'bachelor_thesis/models/dense_char_transformer.pt'
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model.load_state_dict(torch.load(model_path))
    else:
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
            num_epochs=100,  # Full training run
            learning_rate=1e-4,
            weight_decay=0.1,
            warmup_steps=1000,
            device=device,
            patience=8,  # Increased patience for longer training
            min_lr=1e-5,  # Minimum learning rate
            gradient_accumulation_steps=4,  # Gradient accumulation for stability
            use_mixed_precision=True,  # Use mixed precision training
            use_cosine_schedule=True  # Use cosine learning rate schedule
        )
        
        # Save model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        print(f"Saving model to {model_path}")
        torch.save(model.state_dict(), model_path)
    
    # Generate some example text
    print("\nGenerating example texts with different temperatures:")
    prompt = "The quick brown fox"
    
    print("\nConservative sampling (temperature=0.6):")
    generated = generate_text(model, tokenizer, prompt, temperature=0.6, max_length=200)
    print(generated)
    
    print("\nBalanced sampling (temperature=0.8):")
    generated = generate_text(model, tokenizer, prompt, temperature=0.8, max_length=200)
    print(generated)
    
    print("\nCreative sampling (temperature=1.0):")
    generated = generate_text(model, tokenizer, prompt, temperature=1.0, max_length=200)
    print(generated)
    
    # Interactive generation
    print("\nEnter prompts for text generation (type 'exit' to quit):")
    while True:
        prompt = input("\nPrompt: ")
        if prompt.lower() == 'exit':
            break
            
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

if __name__ == "__main__":
    main() 