from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
import os
import argparse
from bpe_tokenizer import BPETokenizer

# 1. Initialize a BPE tokenizer
tokenizer = Tokenizer(models.BPE())

# 2. Set pre-tokenizer (splits input into words)
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# 3. Define a trainer
trainer = trainers.BpeTrainer(
    vocab_size=30000,  # You can change this depending on your needs
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)

# 4. Train tokenizer
tokenizer.train(files='data/enwik8', trainer=trainer)

# 5. (Optional) Set post-processor and decoder
tokenizer.post_processor = processors.TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[("[CLS]", tokenizer.token_to_id("[CLS]")),
                    ("[SEP]", tokenizer.token_to_id("[SEP]"))]
)
tokenizer.decoder = decoders.BPEDecoder()

# 6. Save tokenizer
tokenizer.save("bpe-enwik8-tokenizer.json")
print("Tokenizer trained and saved.")
# Reload and test
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("bpe-enwik8-tokenizer.json")

text = "Byte Pair Encoding is awesome!"
output = tokenizer.encode(text)

print("Tokens:", output.tokens)
print("Token IDs:", output.ids)

def load_training_data(data_path: str, sample_size: int = None) -> str:
    """Load training data for tokenizer"""
    print(f"Loading training data from {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    if sample_size and len(text) > sample_size:
        print(f"Sampling {sample_size:,} characters from {len(text):,} total")
        # Take a continuous chunk from a random position
        import random
        start = random.randint(0, len(text) - sample_size)
        text = text[start:start + sample_size]
    
    return text

def analyze_tokenizer(tokenizer: BPETokenizer, text: str, num_samples: int = 5):
    """Analyze tokenizer performance on sample text"""
    print("\n=== Tokenizer Analysis ===")
    
    # 1. Vocabulary statistics
    print("\nVocabulary Statistics:")
    print(f"Total vocabulary size: {len(tokenizer.encoder):,}")
    print(f"Special tokens: {list(tokenizer.special_tokens.keys())}")
    print(f"Maximum token length: {tokenizer.max_token_length}")
    
    # 2. Sample tokenization
    print("\nSample Tokenizations:")
    # Take a few random sentences from the text
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
    if sentences:
        import random
        samples = random.sample(sentences, min(num_samples, len(sentences)))
        for i, sample in enumerate(samples, 1):
            tokens = tokenizer.encode(sample)
            decoded = tokenizer.decode(tokens)
            print(f"\nSample {i}:")
            print(f"Original : {sample}")
            print(f"Tokenized: {[tokenizer.decoder[t] for t in tokens]}")
            print(f"Decoded  : {decoded}")
            print(f"Tokens   : {len(tokens)}")
    
    # 3. Token length distribution
    token_lengths = [len(token) for token in tokenizer.encoder.keys() 
                    if token not in tokenizer.special_tokens]
    if token_lengths:
        avg_len = sum(token_lengths) / len(token_lengths)
        max_len = max(token_lengths)
        print(f"\nToken Length Statistics:")
        print(f"Average token length: {avg_len:.1f} characters")
        print(f"Maximum token length: {max_len} characters")
        
        # Length distribution
        length_dist = {}
        for length in token_lengths:
            length_dist[length] = length_dist.get(length, 0) + 1
        print("\nToken Length Distribution:")
        for length in sorted(length_dist.keys()):
            count = length_dist[length]
            percentage = count / len(token_lengths) * 100
            print(f"Length {length}: {count:,} tokens ({percentage:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description='Train BPE tokenizer')
    parser.add_argument('--data', type=str, required=True,
                      help='Path to training data file')
    parser.add_argument('--vocab-size', type=int, default=8000,
                      help='Target vocabulary size (default: 8000)')
    parser.add_argument('--output', type=str, default='models/tokenizer.json',
                      help='Output path for trained tokenizer (default: models/tokenizer.json)')
    parser.add_argument('--sample-size', type=int, default=None,
                      help='Number of characters to sample for training (default: use all)')
    parser.add_argument('--min-frequency', type=int, default=1,
                      help='Minimum frequency for merging pairs (default: 1)')
    parser.add_argument('--batch-size', type=int, default=1000000,
                      help='Batch size for processing (default: 1M chars)')
    
    args = parser.parse_args()
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # Load training data
    text = load_training_data(args.data, args.sample_size)
    print(f"Training data size: {len(text):,} characters")
    
    # Initialize and train tokenizer
    print(f"\nInitializing tokenizer with vocabulary size {args.vocab_size:,}")
    tokenizer = BPETokenizer(vocab_size=args.vocab_size)
    
    print("\nTraining tokenizer...")
    tokenizer.train(
        text=text,
        min_frequency=args.min_frequency,
        batch_size=args.batch_size
    )
    
    # Save trained tokenizer
    print(f"\nSaving tokenizer to {args.output}")
    tokenizer.save(args.output)
    
    # Analyze tokenizer performance
    analyze_tokenizer(tokenizer, text)
    
    # Test tokenizer
    print("\nTesting tokenizer on a sample text...")
    sample_text = text[:1000]  # Test on first 1000 chars
    tokens = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(tokens)
    
    print(f"Sample text encoded to {len(tokens)} tokens")
    print(f"Original length: {len(sample_text)} chars")
    print(f"Decoded length: {len(decoded)} chars")
    print(f"Compression ratio: {len(tokens) / len(sample_text):.2f} tokens/char")
    
    # Print a few example tokens
    print("\nExample tokens from sample text:")
    token_freqs = {}
    for token_id in tokens[:100]:  # Look at first 100 tokens
        token = tokenizer.decoder[token_id]
        token_freqs[token] = token_freqs.get(token, 0) + 1
    
    # Print top 10 most frequent tokens
    print("\nTop 10 most frequent tokens in sample:")
    for token, freq in sorted(token_freqs.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"'{token}': {freq} times")

if __name__ == '__main__':
    main()
