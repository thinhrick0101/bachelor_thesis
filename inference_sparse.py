import torch
import logging
from pathlib import Path
from sparse_transformer_v2 import SparseTransformer
from stable_char_transformer import ByteTokenizer
import argparse
import random

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(message)s',
    level=logging.INFO
)

def load_model(model_path, device):
    """Load the trained sparse transformer model."""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model configuration from checkpoint if available
    config = checkpoint.get('model_config', {
        'vocab_size': 256,  # Fixed for byte-level
        'd_model': 512,
        'nhead': 8,
        'num_layers': 6,
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'activation': "gelu",
        'max_seq_length': 1024
    })
    
    # Create model with configuration
    model = SparseTransformer(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        activation=config['activation'],
        max_seq_length=config['max_seq_length']
    )
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model

def generate_text(model, tokenizer, prompt, max_length=512, temperature=0.85, top_k=50, repetition_penalty=1.15, device='cuda'):
    """Generate text using the trained model with ASCII-only output."""
    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    
    # Convert to tensor and handle cloning properly
    input_tensor = input_ids.clone().detach().unsqueeze(0).to(device)  # [1, seq_len]
    
    generated = list(input_ids.cpu().numpy())
    
    # Create causal mask for autoregressive generation
    def create_causal_mask(size):
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask.to(device)
    
    # Function to detect repetition with smaller window
    def is_repetitive(tokens, window_size=3):  # Reduced window size
        if len(tokens) < window_size * 2:
            return False
        last_window = tokens[-window_size:]
        prev_window = tokens[-2*window_size:-window_size]
        return last_window == prev_window
    
    # Function to compute token probabilities with penalties
    def get_next_token_probs(logits, prev_tokens):
        # Apply temperature
        logits = logits / temperature
        
        # Restrict to printable ASCII range (32-126)
        logits[:32] = float('-inf')  # Control characters
        logits[127:] = float('-inf')  # Extended ASCII and Unicode
        
        # Apply repetition penalty more gently
        if len(prev_tokens) > 0:
            # Look at last 10 tokens for repetition
            for token in set(prev_tokens[-10:]):
                logits[token] /= repetition_penalty
        
        # Apply top-k filtering
        if top_k > 0:
            top_k_scores, _ = torch.topk(logits, top_k)
            min_score = top_k_scores[-1]
            logits[logits < min_score] = float('-inf')
        
        # Convert to probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Add small noise for diversity
        noise = torch.randn_like(probs) * 0.01
        probs = probs + noise
        probs = torch.clamp(probs, min=0.0)
        probs = probs / probs.sum()  # Renormalize
        
        return probs
    
    model.eval()
    with torch.no_grad():
        for _ in range(max_length - len(input_ids)):
            # Get current sequence
            if len(generated) > model.max_seq_length:
                current_ids = generated[-model.max_seq_length:]
                input_tensor = torch.tensor(current_ids, dtype=torch.long).unsqueeze(0).to(device)
            else:
                current_ids = generated
                input_tensor = torch.tensor(current_ids, dtype=torch.long).unsqueeze(0).to(device)
            
            # Create attention mask
            seq_len = input_tensor.size(1)
            src_mask = create_causal_mask(seq_len)
            
            # Forward pass with mask
            outputs = model(input_tensor, src_mask=src_mask)
            next_token_logits = outputs[0, -1, :]
            
            # Get probabilities with penalties
            probs = get_next_token_probs(next_token_logits, generated)
            
            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            # Ensure token is in printable ASCII range
            if not (32 <= next_token <= 126):
                # Try sampling again
                probs = get_next_token_probs(next_token_logits, generated)
                next_token = torch.multinomial(probs, num_samples=1).item()
                if not (32 <= next_token <= 126):
                    # If still not printable ASCII, use a space
                    next_token = 32
            
            # Add the token to the sequence
            generated.append(next_token)
            
            # Check for repetition
            if is_repetitive(generated):
                # Try sampling again with higher temperature
                generated.pop()
                retry_temp = temperature * 1.2
                next_token_logits = outputs[0, -1, :] / retry_temp
                probs = get_next_token_probs(next_token_logits, generated)
                next_token = torch.multinomial(probs, num_samples=1).item()
                generated.append(next_token)
                
                # If still repetitive, stop generation
                if is_repetitive(generated):
                    break
            
            # Stop if we generate sentence-ending punctuation and have generated enough tokens
            if (next_token in [ord('.'), ord('!'), ord('?')] and 
                len(generated) > len(input_ids) + 5):  # At least 5 tokens after prompt
                break
            
            # Print progress for long generations
            if len(generated) % 100 == 0:
                logging.info(f"Generated {len(generated)} tokens...")
                # Print partial text
                partial_text = tokenizer.decode(bytes(generated))
                print("\nPartial generation:")
                print("-" * 50)
                print(partial_text)
                print("-" * 50)
    
    return tokenizer.decode(bytes(generated))

def main():
    parser = argparse.ArgumentParser(description='Inference with trained sparse transformer')
    parser.add_argument('--model_path', type=str, default='models/sparse_transformer/checkpoint_epoch_002.pt',
                      help='Path to the trained model checkpoint')
    parser.add_argument('--prompt', type=str, required=True,
                      help='Text prompt to start generation')
    parser.add_argument('--max_length', type=int, default=512,  # Reduced to match training
                      help='Maximum length of generated text')
    parser.add_argument('--temperature', type=float, default=0.85,  # Slightly increased for more diversity
                      help='Sampling temperature (higher = more random)')
    parser.add_argument('--top_k', type=int, default=50,  # Increased for more options
                      help='Top-k sampling parameter (0 = disabled)')
    parser.add_argument('--repetition_penalty', type=float, default=1.15,  # Reduced to be less aggressive
                      help='Penalty for repeating tokens (1.0 = disabled)')
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Load model and tokenizer
    model = load_model(args.model_path, device)
    tokenizer = ByteTokenizer()
    
    # Generate text
    logging.info("Generating text...")
    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        device=device
    )
    
    print("\nGenerated text:")
    print("-" * 50)
    print(generated_text)
    print("-" * 50)

if __name__ == "__main__":
    main() 