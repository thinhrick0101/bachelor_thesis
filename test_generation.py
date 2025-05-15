import torch
import os
import sys
import gzip
import urllib.request
from use_model import load_pretrained_model, generate_text

class CharacterTokenizer:
    """
    Simple character-level tokenizer
    """
    def __init__(self):
        # Initialize with fixed vocab size from training
        self.vocab_size = 2451
        
        # Create basic ASCII mappings
        chars = [chr(i) for i in range(32, 127)] + ['\n']
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        
        print(f"Using vocabulary size: {self.vocab_size} characters")

    def encode(self, text):
        """Convert text to a list of integers, replacing unknown chars with space"""
        return [self.char_to_idx.get(ch, self.char_to_idx[' ']) for ch in text]

    def decode(self, indices):
        """Convert a list of integers to text, handling out-of-vocab indices"""
        return ''.join([self.idx_to_char.get(idx % len(self.idx_to_char), ' ') for idx in indices])

def load_data(data_path, data_url=None):
    """
    Load text data from file or download if not available
    """
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    # Download data if not available
    if not os.path.exists(data_path) and data_url:
        print(f"Downloading data from {data_url}")
        urllib.request.urlretrieve(data_url, data_path + '.gz')

        # Decompress .gz file
        with gzip.open(data_path + '.gz', 'rb') as f_in:
            with open(data_path, 'wb') as f_out:
                f_out.write(f_in.read())

    # Load data
    print(f"Loading data from {data_path}")
    with open(data_path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()

    return text

def test_model(prompt, max_length=256):
    # Initialize tokenizer
    tokenizer = CharacterTokenizer()
    
    # Load model with correct architecture parameters
    model_path = "enhanced_char_transformer_model.pt"
    model, device = load_pretrained_model(
        model_path, 
        vocab_size=2451,      # Use fixed vocab size from training
        d_model=768,          # From training config
        nhead=12,             # From training config
        num_layers=16,        # From training config
        dim_feedforward=3072, # From training config
        dropout=0.2,          # From training config
        attention_dropout=0.15,  # From training config
        activation_dropout=0.15, # From training config
        token_dropout=0.1,      # From training config
        stochastic_depth_prob=0.1  # From training config
    )
    
    # Generate text
    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_length=max_length,
        temperature=0.7,  # You can adjust these parameters
        top_k=20,        # to control the generation
        top_p=0.9,
        repetition_penalty=1.2,
        device=device
    )
    
    return generated_text

if __name__ == "__main__":
    # Get prompt from user
    prompt = input("Enter your prompt: ")
    max_length = input("Enter maximum length (default 150, press Enter to skip): ")
    
    # Set default max_length if empty
    max_length = int(max_length) if max_length.strip() else 150
    
    print("\nGenerating text...\n")
    print("=" * 50)
    print(f"Prompt: {prompt}")
    print("=" * 50)
    
    generated = test_model(prompt, max_length)
    print(f"\nGenerated text:\n{generated}") 