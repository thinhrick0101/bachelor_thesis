import torch
import os
import sys
import traceback

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from stable_char_transformer import EnhancedCharTransformer, ByteTokenizer

def load_model(model_path, device='cuda'):
    """Load the trained model"""
    print(f"\nAttempting to load model from: {model_path}")
    print(f"File exists: {os.path.exists(model_path)}")
    print(f"File size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
    
    # Model configuration - must match the training configuration
    config = {
        'vocab_size': 256,  # ByteTokenizer uses 256 for byte-level tokenization
        'd_model': 512,
        'nhead': 8,
        'num_layers': 12,
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'attention_dropout': 0.1,
        'activation_dropout': 0.1,
        'token_dropout': 0.05,
        'use_checkpoint': True,
        'stochastic_depth_prob': 0.1
    }
    
    try:
        # Create model instance
        print("Creating model instance...")
        model = EnhancedCharTransformer(**config)
        
        # Load the trained weights
        print("Loading state dict...")
        state_dict = torch.load(model_path, map_location=device)
        print("State dict loaded successfully")
        
        model.load_state_dict(state_dict)
        print("State dict applied to model")
        
        model = model.to(device)
        model.eval()
        
        return model
    except Exception as e:
        print(f"\nError during model loading:")
        print(traceback.format_exc())
        raise e

def generate_text(model, tokenizer, prompt, max_length=200, temperature=0.7, top_k=50, top_p=0.9, device='cuda'):
    """Generate text using the trained model"""
    try:
        model.eval()
        print(f"\nGenerating with parameters:")
        print(f"Temperature: {temperature}")
        print(f"Max length: {max_length}")
        print(f"Top k: {top_k}")
        print(f"Top p: {top_p}")
        
        with torch.no_grad():
            # Generate text
            generated = model.generate(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                tokenizer=tokenizer,
                device=device
            )
        
        return generated
    except Exception as e:
        print(f"\nError during text generation:")
        print(traceback.format_exc())
        raise e

def main():
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Initialize tokenizer
        print("Initializing tokenizer...")
        tokenizer = ByteTokenizer()
        print("Tokenizer initialized")
        
        # Load the model - using absolute path
        model_path = 'bachelor_thesis/models/dense_char_transformer.pt'
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return
            
        model = load_model(model_path, device)
        print("Model loaded successfully!")
        
        # Interactive generation loop
        print("\nEnter prompts for text generation. Type 'exit' to quit.")
        while True:
            prompt = input("\nEnter your prompt: ")
            if prompt.lower() == 'exit':
                break
            
            try:
                # Get generation parameters
                temp = float(input("Temperature (0.1-1.0, default 0.7): ") or 0.7)
                length = int(input("Maximum length (default 200): ") or 200)
                
                print("\nGenerating text...")
                generated_text = generate_text(
                    model,
                    tokenizer,
                    prompt,
                    max_length=length,
                    temperature=temp,
                    device=device
                )
                
                print("\nGenerated text:")
                print("-" * 50)
                print(generated_text)
                print("-" * 50)
                
            except Exception as e:
                print(f"Error during generation: {str(e)}")
                print("Please try again with different parameters.")
    
    except Exception as e:
        print(f"\nFatal error:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 