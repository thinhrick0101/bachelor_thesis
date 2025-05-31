import torch
import logging
from pathlib import Path
from sparse_transformer_v2 import SparseTransformer
from stable_char_transformer import ByteTokenizer
import argparse
import random
import torch.nn.functional as F

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

def generate_text(model, tokenizer, prompt, max_length=512, temperature=0.8, top_k=40, top_p=0.9,
                repetition_penalty=1.2, device='cuda'):
    """Generate text using the trained model with proper attention masking."""
    model.eval()
    
    # Convert model to half precision
    model = model.half()
    
    # Encode prompt
    prompt_ids = tokenizer.encode(prompt)
    if isinstance(prompt_ids, torch.Tensor):
        input_ids = prompt_ids.long().unsqueeze(0).to(device)
    else:
        input_ids = torch.from_numpy(prompt_ids).long().unsqueeze(0).to(device)
    
    # Pre-allocate tensors
    generated = []
    past_tokens = set()
    max_len = min(max_length, model.max_seq_length)
    
    # Create simple causal mask - the sparse attention patterns will be handled by the model
    src_mask = torch.triu(torch.ones((max_len, max_len), device=device), diagonal=1).bool()
    
    # Get vocab size from tokenizer or default to 256 for byte-level
    vocab_size = getattr(tokenizer, 'vocab_size', 256)
    
    # Pre-compute penalty mask
    penalty_mask = torch.ones((1, vocab_size), device=device, dtype=torch.float16)
    
    # Track repeated tokens for quality control
    last_tokens = []
    max_repeat_ngram = 5
    repetition_threshold = 3
    repetition_window = 10
    
    with torch.inference_mode():
        for i in range(max_length):
            if input_ids.size(1) > model.max_seq_length:
                input_ids = input_ids[:, -model.max_seq_length:]
            
            seq_len = input_ids.size(1)
            current_mask = src_mask[:seq_len, :seq_len]
            
            try:
                # Forward pass in half precision
                outputs = model(input_ids, src_mask=current_mask)
                logits = outputs[:, -1, :].float()  # Only convert last token to float32
                
                # Temperature scaling with dynamic adjustment
                if len(last_tokens) >= repetition_window:
                    unique_ratio = len(set(last_tokens[-repetition_window:])) / repetition_window
                    # Increase temperature if too repetitive
                    adjusted_temperature = temperature * (1.0 + (1.0 - unique_ratio))
                else:
                    adjusted_temperature = temperature
                logits.div_(adjusted_temperature)
                
                # Enhanced repetition penalty
                if past_tokens:
                    penalty_mask.fill_(1)
                    # Apply stronger penalty for recently used tokens
                    recent_tokens = list(past_tokens)[-20:]  # Last 20 tokens
                    penalty_values = torch.tensor([repetition_penalty * (1.1 ** (len(recent_tokens) - i)) 
                                                for i, _ in enumerate(recent_tokens)], device=device)
                    penalty_mask.index_fill_(1, torch.tensor(recent_tokens, device=device), 
                                          penalty_values.unsqueeze(0))
                    logits.div_(penalty_mask)
                
                # Dynamic top-k based on sequence length
                current_top_k = max(2, min(top_k, vocab_size // 2))
                if top_k > 0:
                    v, _ = torch.topk(logits, current_top_k)
                    logits[logits < v[:, [-1]]] = -float('inf')
                
                # Nucleus sampling with dynamic threshold
                if top_p < 1.0:
                    probs = torch.softmax(logits, dim=-1)
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                    # Adjust top_p based on sequence length
                    dynamic_top_p = max(0.1, top_p * (1.0 - (seq_len / max_length) * 0.2))
                    mask = cumsum_probs > dynamic_top_p
                    mask[..., 1:] = mask[..., :-1].clone()
                    mask[..., 0] = 0
                    probs.scatter_(1, sorted_indices, probs.gather(1, sorted_indices) * (~mask).float())
                    probs.div_(probs.sum(dim=-1, keepdim=True))
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                
                token_id = next_token.item()
                
                # Check for excessive repetition
                last_tokens.append(token_id)
                if len(last_tokens) > repetition_window:
                    last_tokens.pop(0)
                    if len(set(last_tokens)) < repetition_window / repetition_threshold:
                        # Try to break repetition by increasing temperature
                        continue
                
                # Check for repeated n-grams
                if len(generated) >= max_repeat_ngram:
                    current_ngram = generated[-(max_repeat_ngram-1):] + [token_id]
                    found_repeat = False
                    for j in range(len(generated) - max_repeat_ngram + 1):
                        if generated[j:j+max_repeat_ngram] == current_ngram:
                            found_repeat = True
                            break
                    if found_repeat:
                        continue
                
                # Update sequences
                input_ids = torch.cat([input_ids, next_token], dim=1)
                generated.append(token_id)
                past_tokens.add(token_id)
                
                if token_id in [tokenizer.eos_token_id] if hasattr(tokenizer, 'eos_token_id') else []:
                    break
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    if i > 0:
                        break
                raise e
    
    try:
        return tokenizer.decode(generated)
    except Exception as e:
        logging.error(f"Error decoding generated tokens: {e}")
        return "[DECODING_ERROR]"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default='The movie was', help='Input prompt')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--max_length', type=int, default=1024, help='Maximum length to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=40, help='Top-k filtering value')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p (nucleus) filtering value')
    parser.add_argument('--repetition_penalty', type=float, default=1.2, help='Repetition penalty')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Load model
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        model_config = checkpoint['model_config']
        model = SparseTransformer(**model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        # Log model configuration
        logging.info("Model configuration:")
        for key, value in model_config.items():
            logging.info(f"  {key}: {value}")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return
    
    # Create tokenizer
    tokenizer = ByteTokenizer()
    
    # Generate text
    logging.info("Generating text...")
    try:
        generated_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            device=device
        )
        print(f"\nPrompt: {args.prompt}")
        print(f"Generated: {generated_text}")
    except Exception as e:
        logging.error(f"Error during text generation: {e}")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 