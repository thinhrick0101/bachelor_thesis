import os
import json
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm  # For progress tracking

class BPETokenizer:
    """
    A simplified implementation of Byte-Pair Encoding (BPE) tokenizer
    with optimized training process
    """
    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.encoder: Dict[str, int] = {}
        self.decoder: Dict[int, str] = {}
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1,
            '<s>': 2,
            '</s>': 3,
            '<mask>': 4
        }
        self.max_token_length = 0  # Track maximum token length for optimization
    
    def _init_vocab(self, text: str) -> None:
        """Initialize vocabulary with all unique characters and special tokens"""
        # Add special tokens
        self.encoder = self.special_tokens.copy()
        self.decoder = {v: k for k, v in self.encoder.items()}
        next_id = len(self.encoder)
        
        # Add all unique characters from text
        chars = sorted(set(text))
        for c in chars:
            if c not in self.encoder:
                self.encoder[c] = next_id
                self.decoder[next_id] = c
                next_id += 1
        
        self.max_token_length = 1
        print(f"Initial vocabulary size: {len(self.encoder)} tokens")
    
    def _get_stats(self, ids: List[List[int]], min_frequency: int) -> Counter:
        """Get counts of adjacent pairs with optimized counting"""
        pairs = Counter()
        
        # Process each piece in parallel for large datasets
        for piece in ids:
            if len(piece) < 2:
                continue
                
            # Count pairs in current piece
            for i in range(len(piece) - 1):
                pair = (piece[i], piece[i + 1])
                pairs[pair] += 1
                
                # Early break if we've found a very frequent pair
                if pairs[pair] >= min_frequency * 10:
                    return pairs
        
        return pairs
    
    def train(self, text: str, min_frequency: int = 2, batch_size: int = 1000000):
        """
        Train the tokenizer on text with optimized processing and progress tracking
        
        Args:
            text: Training text
            min_frequency: Minimum frequency for a pair to be merged
            batch_size: Number of characters to process in each batch
        """
        # Initialize vocabulary with all characters
        self._init_vocab(text)
        
        # Convert text to base tokens in batches
        print("Converting text to base tokens...")
        words = []
        for i in range(0, len(text), batch_size):
            batch = text[i:i + batch_size]
            words.extend([[c for c in word] for word in batch.split()])
        
        word_ids = [[self.encoder[c] for c in word] for word in words]
        
        # Track progress with tqdm
        pbar = tqdm(total=self.vocab_size - len(self.encoder), 
                   desc="Training BPE tokenizer")
        
        last_pair_count = float('inf')
        unchanged_count = 0
        
        while len(self.encoder) < self.vocab_size:
            # Find most common pair
            pairs = self._get_stats(word_ids, min_frequency)
            if not pairs:
                print("\nNo more pairs found above minimum frequency")
                break
            
            # Check for convergence
            max_pair_count = max(pairs.values())
            if max_pair_count == last_pair_count:
                unchanged_count += 1
                if unchanged_count >= 5:
                    print("\nConverged: No significant changes in pair frequencies")
                    break
            else:
                unchanged_count = 0
            last_pair_count = max_pair_count
            
            # Get most frequent pair
            pair = max(pairs.items(), key=lambda x: x[1])
            if pair[1] < min_frequency:
                print("\nNo more pairs above minimum frequency threshold")
                break
            
            # Create new token
            new_token = self.decoder[pair[0][0]] + self.decoder[pair[0][1]]
            self.encoder[new_token] = len(self.encoder)
            self.decoder[len(self.encoder) - 1] = new_token
            
            # Update maximum token length
            self.max_token_length = max(self.max_token_length, len(new_token))
            
            # Merge pairs in the data with optimized processing
            for word_idx, word in enumerate(word_ids):
                i = 0
                while i < len(word) - 1:
                    if (word[i], word[i + 1]) == pair[0]:
                        word[i:i + 2] = [self.encoder[new_token]]
                    else:
                        i += 1
            
            # Update progress bar
            pbar.update(1)
            
            # Print occasional statistics
            if len(self.encoder) % 1000 == 0:
                print(f"\nVocabulary size: {len(self.encoder)}")
                print(f"Most frequent pair: {new_token} (count: {pair[1]})")
                print(f"Maximum token length: {self.max_token_length}")
        
        pbar.close()
        print(f"\nFinal vocabulary size: {len(self.encoder)}")
        print(f"Maximum token length: {self.max_token_length}")
    
    def save(self, path: str):
        """Save tokenizer to file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'encoder': self.encoder,
                'decoder': {str(k): v for k, v in self.decoder.items()},
                'vocab_size': self.vocab_size,
                'special_tokens': self.special_tokens,
                'max_token_length': self.max_token_length
            }, f, ensure_ascii=False, indent=2)
    
    def load(self, path: str):
        """Load tokenizer from file"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.encoder = data['encoder']
            self.decoder = {int(k): v for k, v in data['decoder'].items()}
            self.vocab_size = data['vocab_size']
            self.special_tokens = data['special_tokens']
            self.max_token_length = data.get('max_token_length', 1)
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token ids with optimized processing"""
        if not text:
            return []
        
        tokens = []
        for word in text.split():
            # Try to encode whole word first
            if word in self.encoder:
                tokens.append(self.encoder[word])
                continue
            
            # Try to encode subwords efficiently
            current_token = ""
            i = 0
            while i < len(word):
                found = False
                # Try longest possible tokens first
                for j in range(min(self.max_token_length, len(word) - i), 0, -1):
                    subword = word[i:i+j]
                    if subword in self.encoder:
                        tokens.append(self.encoder[subword])
                        i += j
                        found = True
                        break
                
                if not found:
                    # If no token found, add unknown token
                    tokens.append(self.encoder['<unk>'])
                    i += 1
            
            # Add space between words (if not last word)
            if word != text.split()[-1]:
                tokens.append(self.encoder.get(' ', self.encoder['<unk>']))
        
        return tokens
    
    def decode(self, ids: List[int]) -> str:
        """Decode token ids to text with improved handling of special tokens"""
        text = []
        for id in ids:
            token = self.decoder.get(id, '<unk>')
            # Skip special tokens in output
            if token not in self.special_tokens:
                text.append(token)
        return ''.join(text) 