from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
import os
import argparse

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

def analyze_tokenizer(tokenizer: Tokenizer, text: str, num_samples: int = 5):
    """Analyze tokenizer performance on sample text"""
    print("\n=== Tokenizer Analysis ===")
    
    # 1. Vocabulary statistics
    vocab = tokenizer.get_vocab()
    print("\nVocabulary Statistics:")
    print(f"Total vocabulary size: {len(vocab):,}")
    print(f"Special tokens: {[token for token, _ in sorted(vocab.items()) if token.startswith('[') and token.endswith(']')]}")
    
    # 2. Sample tokenization
    print("\nSample Tokenizations:")
    # Take a few random sentences from the text
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
    if sentences:
        import random
        samples = random.sample(sentences, min(num_samples, len(sentences)))
        for i, sample in enumerate(samples, 1):
            output = tokenizer.encode(sample)
            decoded = tokenizer.decode(output.ids)
            print(f"\nSample {i}:")
            print(f"Original : {sample}")
            print(f"Tokenized: {output.tokens}")
            print(f"Decoded  : {decoded}")
            print(f"Tokens   : {len(output.tokens)}")
    
    # 3. Token length distribution
    token_lengths = [len(token) for token in vocab.keys() 
                    if not (token.startswith('[') and token.endswith(']'))]
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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train BPE tokenizer using Hugging Face tokenizers')
    parser.add_argument('--data', type=str, default='data/enwik8',
                      help='Path to training data file (default: data/enwik8)')
    parser.add_argument('--vocab-size', type=int, default=30000,
                      help='Target vocabulary size (default: 30000)')
    parser.add_argument('--output', type=str, default='bpe-enwik8-tokenizer.json',
                      help='Output path for trained tokenizer (default: bpe-enwik8-tokenizer.json)')
    parser.add_argument('--min-frequency', type=int, default=2,
                      help='Minimum frequency for merging (default: 2)')
    
    args = parser.parse_args()

    # 1. Initialize a BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    # 2. Set pre-tokenizer (splits input into words)
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # 3. Define a trainer with command line parameters
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        show_progress=True
    )

    # 4. Train tokenizer
    print(f"Training tokenizer on {args.data}")
    print(f"Target vocabulary size: {args.vocab_size}")
    print(f"Minimum merge frequency: {args.min_frequency}")
    tokenizer.train(files=[args.data], trainer=trainer)

    # 5. Set post-processor and decoder
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", tokenizer.token_to_id("[CLS]")),
                      ("[SEP]", tokenizer.token_to_id("[SEP]"))]
    )
    tokenizer.decoder = decoders.BPEDecoder()

    # 6. Save tokenizer
    tokenizer.save(args.output)
    print(f"\nTokenizer trained and saved to {args.output}")

    # 7. Test the tokenizer
    print("\nTesting tokenizer on sample texts:")
    test_texts = [
        "Byte Pair Encoding is awesome!",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models need good tokenization."
    ]

    # Reload the tokenizer to ensure it was saved correctly
    tokenizer = Tokenizer.from_file(args.output)
    
    for text in test_texts:
        output = tokenizer.encode(text)
        print(f"\nInput    : {text}")
        print(f"Tokens   : {output.tokens}")
        print(f"Token IDs: {output.ids}")

    # Print vocabulary statistics
    print(f"\nFinal vocabulary size: {tokenizer.get_vocab_size():,}")
    print(f"Number of special tokens: {len(['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'])}")

if __name__ == "__main__":
    main()
