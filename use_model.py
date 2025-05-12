import torch
import os
import sys
from torch.nn import functional as F

# Add current directory to path to ensure we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the model classes
from stable_char_transformer import EnhancedCharTransformer, CharacterTokenizer

# Define a simplified tokenizer we can use without the original training data
class SimpleCharTokenizer:
    """Simplified character tokenizer with basic ASCII characters"""
    def __init__(self):
        # Create a simple character set (printable ASCII)
        chars = [chr(i) for i in range(32, 127)] + ['\n']
        self.vocab_size = len(chars)
        
        # Create mappings
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        
        print(f"Vocabulary size: {self.vocab_size} characters")
    
    def encode(self, text):
        """Convert text to a list of integers, replacing unknown chars with space"""
        return [self.char_to_idx.get(ch, self.char_to_idx[' ']) for ch in text]
    
    def decode(self, indices):
        """Convert a list of integers to text"""
        return ''.join([self.idx_to_char.get(idx, ' ') for idx in indices])

def load_pretrained_model(model_path, vocab_size=96):
    """Load the pre-trained model with the correct architecture"""
    # These parameters should match those used during training
    # Using common values for transformer models of this size
    model = EnhancedCharTransformer(
        vocab_size=vocab_size,
        d_model=512,             # Embedding dimension
        nhead=8,                 # Number of attention heads
        num_layers=6,            # Number of transformer layers
        dim_feedforward=2048,    # Feedforward network dimension
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.1,
        token_dropout=0.05,
        use_checkpoint=False     # No need for checkpointing during inference
    )
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set to evaluation mode
    model.to(device)
    
    return model, device

def generate_text(model, tokenizer, prompt, max_length=200, temperature=0.7, 
                 top_k=20, top_p=0.9, repetition_penalty=1.2, device='cpu'):
    """Generate text using the model"""
    # Check if the model has a generate method
    if hasattr(model, 'generate'):
        # Use the model's built-in generate method
        return model.generate(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            tokenizer=tokenizer,
            repetition_penalty=repetition_penalty,
            device=device
        )
    else:
        # Implement generation manually if needed
        # Convert prompt to tensor
        prompt_tensor = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
        
        # Generate text
        generated = prompt_tensor.clone()
        
        # Set model to eval mode
        model.eval()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                outputs = model(generated)
                
                # Get the next token predictions
                next_token_logits = outputs[0, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply repetition penalty
                for token_id in set(generated[0].tolist()):
                    next_token_logits[token_id] /= repetition_penalty
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][-1]
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # Sample from the filtered distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Add the token to the generated sequence
                generated = torch.cat((generated, next_token), dim=1)
                
                # Stop generation if we reach a newline or end of text token
                if next_token.item() == tokenizer.char_to_idx.get('\n', -1):
                    break
        
        # Decode and return the generated text
        return tokenizer.decode(generated[0].tolist())

def main():
    # Path to the pre-trained model
    model_path = "stable_char_transformer_model.pt"
    
    # Initialize tokenizer
    tokenizer = SimpleCharTokenizer()
    
    # Load model
    model, device = load_pretrained_model(model_path, vocab_size=tokenizer.vocab_size)
    
    # Generate text with different prompts
    prompts = [
        "Once upon a time",
        "The weather today is",
        "In the future, AI will",
        "The most important thing to remember is"
    ]
    
    for prompt in prompts:
        print("\n" + "="*50)
        print(f"Prompt: {prompt}")
        print("="*50)
        
        generated_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_length=150,
            temperature=0.7,
            top_k=20,
            top_p=0.9,
            repetition_penalty=1.2,
            device=device
        )
        
        print(f"Generated:\n{generated_text}")

if __name__ == "__main__":
    main() 