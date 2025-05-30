import torch
import logging
from pathlib import Path
from sparse_transformer_v2 import SparseTransformer
from stable_char_transformer import ByteTokenizer
import argparse

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(message)s',
    level=logging.INFO
)

def load_model(model_path, device):
    """Load the trained sparse transformer model."""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model with same configuration
    model = SparseTransformer(
        vocab_size=256,  # Fixed for byte-level
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="gelu",
        max_seq_length=1024
    )
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model

def generate_text(model, tokenizer, prompt, max_length=1024, temperature=1.0, top_k=50, repetition_penalty=1.2, device='cuda'):
    """Generate text using the trained model."""
    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    
    # Convert to tensor and handle cloning properly
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)  # [1, seq_len]
    
    generated = list(input_ids)
    
    # Create causal mask for autoregressive generation
    def create_causal_mask(size):
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask.to(device)
    
    # Function to detect repetition
    def is_repetitive(tokens, window_size=10):
        if len(tokens) < window_size * 2:
            return False
        last_window = tokens[-window_size:]
        prev_window = tokens[-2*window_size:-window_size]
        return last_window == prev_window
    
    model.eval()
    with torch.no_grad():
        for _ in range(max_length - len(input_ids)):
            # Get current sequence
            if len(generated) > 1024:  # Handle long sequences with sliding window
                current_ids = generated[-1024:]
                input_tensor = torch.tensor(current_ids, dtype=torch.long).unsqueeze(0).to(device)
            else:
                current_ids = generated
                input_tensor = torch.tensor(current_ids, dtype=torch.long).unsqueeze(0).to(device)
            
            # Create attention mask
            seq_len = input_tensor.size(1)
            src_mask = create_causal_mask(seq_len)
            
            # Forward pass with mask
            outputs = model(input_tensor, src_mask=src_mask)
            next_token_logits = outputs[0, -1, :] / temperature
            
            # Apply repetition penalty
            if len(generated) > 0:
                for token in set(generated[-20:]):  # Look at last 20 tokens
                    next_token_logits[token] /= repetition_penalty
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample from the filtered distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            # Check for excessive repetition
            generated.append(next_token)
            if is_repetitive(generated):
                generated = generated[:-10]  # Remove last repetitive sequence
                break
            
            # Stop if we generate a newline sequence or reach max length
            if len(generated) >= 3 and generated[-2:] == [ord('\n'), ord('\n')]:
                break
            
            # Print progress for long generations
            if len(generated) % 100 == 0:
                logging.info(f"Generated {len(generated)} tokens...")
                # Print partial text
                partial_text = tokenizer.decode(generated)
                print("\nPartial generation:")
                print("-" * 50)
                print(partial_text)
                print("-" * 50)
    
    return tokenizer.decode(generated)

def main():
    parser = argparse.ArgumentParser(description='Inference with trained sparse transformer')
    parser.add_argument('--model_path', type=str, default='models/sparse_transformer/best_model.pt',
                      help='Path to the trained model checkpoint')
    parser.add_argument('--prompt', type=str, required=True,
                      help='Text prompt to start generation')
    parser.add_argument('--max_length', type=int, default=1024,
                      help='Maximum length of generated text')
    parser.add_argument('--temperature', type=float, default=1.2,  # Increased from 0.8
                      help='Sampling temperature (higher = more random)')
    parser.add_argument('--top_k', type=int, default=40,
                      help='Top-k sampling parameter (0 = disabled)')
    parser.add_argument('--repetition_penalty', type=float, default=1.2,
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