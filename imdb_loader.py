import os
import torch
import numpy as np
import random
import urllib.request
import tarfile
from collections import Counter
import re

def download_imdb():
    """
    Download and extract the IMDb dataset if it doesn't exist
    """
    url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    filename = 'aclImdb_v1.tar.gz'
    extract_dir = 'aclImdb'

    # Check if the dataset directory already exists
    if os.path.exists(extract_dir):
        print(f"IMDb dataset already exists at {extract_dir}")
        return

    # Download the dataset
    print(f"Downloading IMDb dataset from {url}...")
    urllib.request.urlretrieve(url, filename)

    # Extract the dataset
    print(f"Extracting {filename}...")
    with tarfile.open(filename, 'r:gz') as tar:
        tar.extractall()

    print("IMDb dataset downloaded and extracted successfully.")

def load_imdb_data(final=False, val_size=5000, use_cache=True):
    """
    Load the IMDb dataset

    Args:
        final: If True, use the test set for evaluation. If False, use a validation split.
        val_size: Size of the validation set if final=False
        use_cache: If True, try to load preprocessed data from cache

    Returns:
        (x_train, y_train), (x_val, y_val), (i2w, w2i), num_classes
    """
    import os
    import pickle

    # Define cache file paths
    cache_dir = 'imdb_cache'
    os.makedirs(cache_dir, exist_ok=True)

    cache_file = os.path.join(cache_dir, f'imdb_data_final_{final}_valsize_{val_size}.pkl')

    # Try to load from cache
    if use_cache and os.path.exists(cache_file):
        print(f"Loading preprocessed IMDb data from cache...")
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                print("Cache loaded successfully!")
                if final:
                    return (cache_data['x_train'], cache_data['y_train']), \
                           (cache_data['x_test'], cache_data['y_test']), \
                           (cache_data['i2w'], cache_data['w2i']), 2
                else:
                    return (cache_data['x_train_new'], cache_data['y_train_new']), \
                           (cache_data['x_val'], cache_data['y_val']), \
                           (cache_data['i2w'], cache_data['w2i']), 2
        except Exception as e:
            print(f"Error loading cache: {e}. Processing data from scratch.")

    # Download the dataset if it doesn't exist
    download_imdb()

    # Load the data
    print("Loading IMDb dataset...")
    train_pos = load_files('aclImdb/train/pos')
    train_neg = load_files('aclImdb/train/neg')
    test_pos = load_files('aclImdb/test/pos')
    test_neg = load_files('aclImdb/test/neg')

    # Tokenize the data
    print("Tokenizing data...")
    train_pos_tokens = [tokenize(text) for text in train_pos]
    train_neg_tokens = [tokenize(text) for text in train_neg]
    test_pos_tokens = [tokenize(text) for text in test_pos]
    test_neg_tokens = [tokenize(text) for text in test_neg]

    # Build vocabulary
    print("Building vocabulary...")
    all_tokens = []
    for tokens in train_pos_tokens + train_neg_tokens:
        all_tokens.extend(tokens)

    # Count token frequencies and keep tokens that appear at least 5 times
    counter = Counter(all_tokens)
    vocab = [token for token, count in counter.items() if count >= 5]

    # Create word to index and index to word mappings
    i2w = [".pad", ".start", ".end", ".unk"] + vocab
    w2i = {t: i for i, t in enumerate(i2w)}

    # Convert tokens to indices
    print("Converting tokens to indices...")
    train_pos_indices = [tokens_to_indices(tokens, w2i) for tokens in train_pos_tokens]
    train_neg_indices = [tokens_to_indices(tokens, w2i) for tokens in train_neg_tokens]
    test_pos_indices = [tokens_to_indices(tokens, w2i) for tokens in test_pos_tokens]
    test_neg_indices = [tokens_to_indices(tokens, w2i) for tokens in test_neg_tokens]

    # Combine positive and negative examples
    x_train = train_pos_indices + train_neg_indices
    y_train = [0] * len(train_pos_indices) + [1] * len(train_neg_indices)

    x_test = test_pos_indices + test_neg_indices
    y_test = [0] * len(test_pos_indices) + [1] * len(test_neg_indices)

    # Sort by length
    x_train, y_train = sort_by_len(x_train, y_train)
    x_test, y_test = sort_by_len(x_test, y_test)

    # Prepare return values
    if final:
        result = (x_train, y_train), (x_test, y_test), (i2w, w2i), 2

        # Save to cache
        if use_cache:
            cache_data = {
                'x_train': x_train,
                'y_train': y_train,
                'x_test': x_test,
                'y_test': y_test,
                'i2w': i2w,
                'w2i': w2i
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
                print(f"Saved preprocessed data to {cache_file}")

        return result
    else:
        # Use a validation split
        val_indices = list(range(len(x_train)))[:val_size]
        train_indices = list(range(len(x_train)))[val_size:]

        x_val = [x_train[i] for i in val_indices]
        y_val = [y_train[i] for i in val_indices]

        x_train_new = [x_train[i] for i in train_indices]
        y_train_new = [y_train[i] for i in train_indices]

        result = (x_train_new, y_train_new), (x_val, y_val), (i2w, w2i), 2

        # Save to cache
        if use_cache:
            cache_data = {
                'x_train': x_train,
                'y_train': y_train,
                'x_test': x_test,
                'y_test': y_test,
                'x_train_new': x_train_new,
                'y_train_new': y_train_new,
                'x_val': x_val,
                'y_val': y_val,
                'i2w': i2w,
                'w2i': w2i
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
                print(f"Saved preprocessed data to {cache_file}")

        return result

def load_files(directory):
    """
    Load all text files from a directory
    """
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                texts.append(f.read())
    return texts

def tokenize(text):
    """
    Tokenize text by lowercasing, removing non-letters, and splitting on whitespace
    """
    # Convert to lowercase
    text = text.lower()
    # Replace non-letters with spaces
    text = re.sub(r'[^a-z]', ' ', text)
    # Split on whitespace and filter out empty tokens
    tokens = [token for token in text.split() if token]
    return tokens

def tokens_to_indices(tokens, w2i):
    """
    Convert tokens to indices using the word-to-index mapping
    """
    return [w2i.get(token, w2i['.unk']) for token in tokens]

def sort_by_len(data, labels):
    """
    Sort a dataset by sequence length
    """
    data, labels = zip(*sorted(zip(data, labels), key=lambda p: len(p[0])))
    return list(data), list(labels)

def create_batches(sequences, labels, batch_size, w2i, variable_batch_size=True):
    """
    Create batches of data with variable batch sizes based on sequence length

    Args:
        sequences: List of sequences (each sequence is a list of word indices)
        labels: List of labels for each sequence
        batch_size: Base batch size
        w2i: Word to index mapping dictionary
        variable_batch_size: If True, use larger batch sizes for shorter sequences

    Returns:
        List of (padded_batch, labels_batch) tuples as PyTorch tensors
    """
    # Get the padding token index
    pad_idx = w2i[".pad"]

    # Group sequences by length
    length_groups = {}
    for i, seq in enumerate(sequences):
        length = len(seq)
        if length not in length_groups:
            length_groups[length] = []
        length_groups[length].append((seq, labels[i]))

    # Create batches
    batches = []

    # Process each length group
    for length, group in sorted(length_groups.items()):
        # Calculate batch size for this length group
        if variable_batch_size:
            # Use larger batch sizes for shorter sequences
            # This is a simple heuristic: batch_size * (max_length / length)
            # You can adjust this based on your memory constraints
            adjusted_batch_size = min(batch_size * (1000 / max(length, 1)), len(group))
            adjusted_batch_size = max(1, int(adjusted_batch_size))
        else:
            adjusted_batch_size = batch_size

        # Create batches for this length group
        for i in range(0, len(group), adjusted_batch_size):
            batch_data = group[i:i+adjusted_batch_size]
            batch_sequences = [item[0] for item in batch_data]
            batch_labels = [item[1] for item in batch_data]

            # Pad sequences to the same length
            max_length = max(len(seq) for seq in batch_sequences)
            padded_batch = []
            for seq in batch_sequences:
                padded_seq = seq + [pad_idx] * (max_length - len(seq))
                padded_batch.append(padded_seq)

            # Convert to PyTorch tensors
            padded_batch = torch.tensor(padded_batch, dtype=torch.long)
            labels_batch = torch.tensor(batch_labels, dtype=torch.long)

            batches.append((padded_batch, labels_batch))

    # Shuffle the batches to avoid training on all short or all long sequences consecutively
    random.shuffle(batches)

    return batches

def create_packed_batches(sequences, labels, batch_size, w2i, variable_batch_size=True):
    """
    Create batches of data with packed sequences

    Args:
        sequences: List of sequences (each sequence is a list of word indices)
        labels: List of labels for each sequence
        batch_size: Base batch size
        w2i: Word to index mapping dictionary
        variable_batch_size: If True, use larger batch sizes for shorter sequences

    Returns:
        List of (packed_batch, labels_batch) tuples
    """
    # Group sequences by length
    length_groups = {}
    for i, seq in enumerate(sequences):
        length = len(seq)
        if length not in length_groups:
            length_groups[length] = []
        length_groups[length].append((seq, labels[i]))

    # Create batches
    batches = []

    # Process each length group
    for length, group in sorted(length_groups.items()):
        # Calculate batch size for this length group
        if variable_batch_size:
            # Use larger batch sizes for shorter sequences
            adjusted_batch_size = min(batch_size * (1000 / max(length, 1)), len(group))
            adjusted_batch_size = max(1, int(adjusted_batch_size))
        else:
            adjusted_batch_size = batch_size

        # Create batches for this length group
        for i in range(0, len(group), adjusted_batch_size):
            batch_data = group[i:i+adjusted_batch_size]
            batch_sequences = [torch.tensor(item[0], dtype=torch.long) for item in batch_data]
            batch_labels = [item[1] for item in batch_data]

            # Pack the sequences
            packed_batch = torch.nn.utils.rnn.pack_sequence(batch_sequences, enforce_sorted=False)
            labels_batch = torch.tensor(batch_labels, dtype=torch.long)

            batches.append((packed_batch, labels_batch))

    # Shuffle the batches
    random.shuffle(batches)

    return batches

if __name__ == "__main__":
    # Test the data loading
    (x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = load_imdb_data(final=False)

    print(f"Training examples: {len(x_train)}")
    print(f"Validation examples: {len(x_val)}")
    print(f"Vocabulary size: {len(i2w)}")
    print(f"Number of classes: {numcls}")

    # Test batch creation
    batches = create_batches(x_train[:100], y_train[:100], 16, w2i)
    print(f"Number of batches: {len(batches)}")
    print(f"First batch shape: {batches[0][0].shape}")

    # Test packed batch creation
    packed_batches = create_packed_batches(x_train[:100], y_train[:100], 16, w2i)
    print(f"Number of packed batches: {len(packed_batches)}")
