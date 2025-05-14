import os
from bpe_tokenizer import BPETokenizer

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

def main():
    vocab_size = 8000
    data_path = 'data/enwik8'
    data_url = 'https://codeberg.org/pbm/former/raw/branch/master/data/enwik8.gz'
    print("Loading data...")
    text = load_data(data_path, data_url)
    print(f"Data loaded: {len(text)} characters")
    max_chars = 20000000
    if len(text) > max_chars:
        print(f"Limiting data to first {max_chars} characters for training")
        text = text[:max_chars]

    # Create and train BPE tokenizer
    print("Training BPE tokenizer...")
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(text, min_frequency=2)
    
    # Save tokenizer for later use
    tokenizer.save('data/bpe_tokenizer.json')
    print("Tokenizer saved to data/bpe_tokenizer.json")