import torch
import os
import sys
from torch.nn import functional as F

# Add current directory to path to ensure we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the model classes
from stable_char_transformer import EnhancedCharTransformer

class SimpleCharTokenizer:
    """Character tokenizer that handles the model's vocabulary"""
    def __init__(self):
        # Create a simple character set (printable ASCII)
        chars = [chr(i) for i in range(32, 127)] + ['\n']
        # Add special tokens and extended vocabulary to match the model
        self.vocab_size = 2451  # Match the model's vocabulary size
        
        # Create mappings for basic characters
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        
        print(f"Vocabulary size: {self.vocab_size} characters")
    
    def encode(self, text):
        """Convert text to a list of integers, replacing unknown chars with space"""
        return [self.char_to_idx.get(ch, self.char_to_idx[' ']) for ch in text]
    
    def decode(self, indices):
        """Convert a list of integers to text, ignoring tokens beyond basic ASCII"""
        return ''.join([self.idx_to_char.get(idx, ' ') for idx in indices if idx < len(self.idx_to_char)])

def load_pretrained_model(model_path, vocab_size=2451, d_model=768, nhead=12, 
                         num_layers=16, dim_feedforward=3072, dropout=0.2,
                         attention_dropout=0.15, activation_dropout=0.15,
                         token_dropout=0.1, stochastic_depth_prob=0.1):
    """Load the pre-trained model with the correct architecture"""
    model = EnhancedCharTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,  # This will be doubled by SwiGLU internally
        dropout=dropout,
        attention_dropout=attention_dropout,
        activation_dropout=activation_dropout,
        token_dropout=token_dropout,
        use_checkpoint=False,
        stochastic_depth_prob=stochastic_depth_prob
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