# Set up generation parameters
import torch
from stable_char_transformer import EnhancedCharTransformer, CharacterTokenizer, load_data

# Load text data to build tokenizer
text = load_data("D:/bachelor_thesis/data/enwik8")  # Adjust path as needed
tokenizer = CharacterTokenizer(text)

# Initialize the model architecture (parameters must match the trained model)
model = EnhancedCharTransformer(
    vocab_size=tokenizer.vocab_size,
    d_model=512,         # Embedding dimension
    nhead=8,             # Number of attention heads
    num_layers=6,        # Number of transformer layers
    dim_feedforward=2048 # Feedforward network dimension
)

# Load the pre-trained model weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("stable_char_transformer_model.pt", map_location=device))
model.eval()  # Set to evaluation mode
model.to(device)
prompt = "what is the capital of the Vietnam?"  # Your starting text
max_length = 200             # Maximum length of generated text
temperature = 0.7            # Controls randomness (higher = more random)
top_k = 20                   # Number of highest probability tokens to consider
top_p = 0.9                  # Nucleus sampling threshold

# Generate text
generated_text = model.generate(
    prompt=prompt,
    max_length=max_length,
    temperature=temperature,
    top_k=top_k,
    top_p=top_p,
    tokenizer=tokenizer,
    device=device,
    repetition_penalty=1.2   # Penalizes repeated tokens
)

print(f"Generated text:\n{generated_text}")