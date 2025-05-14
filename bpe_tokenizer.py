import os
import json
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional

class BPETokenizer:
    """
    A simplified implementation of Byte-Pair Encoding (BPE) tokenizer
    """
    def __init__(self, vocab_size: int = 2451):
        self.vocab_size = vocab_size
        self.encoder: Dict[str, int] = {}  # token -> id
        self.decoder: Dict[int, str] = {}  # id -> token
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1,
            '<s>': 2,
            '</s>': 3,
            '<mask>': 4
        }
    
    def _init_vocab(self, text: str) -> None:
        """Initialize vocabulary with all unique characters in text"""
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
    
    def _get_stats(self, ids: List[List[int]]) -> Counter:
        """Get counts of adjacent pairs"""
        pairs = Counter()
        for piece in ids:
            if len(piece) < 2: continue
            for i in range(len(piece) - 1):
                pair = (piece[i], piece[i + 1])
                pairs[pair] += 1
        return pairs
    
    def train(self, text: str, min_frequency: int = 2):
        """Train the tokenizer on text"""
        # Initialize vocabulary with all characters
        self._init_vocab(text)
        
        # Convert text to base tokens
        words = [[c for c in word] for word in text.split()]
        word_ids = [[self.encoder[c] for c in word] for word in words]
        
        while len(self.encoder) < self.vocab_size:
            # Find most common pair
            pairs = self._get_stats(word_ids)
            if not pairs:
                break
                
            pair = max(pairs.items(), key=lambda x: x[1])
            if pair[1] < min_frequency:
                break
                
            # Create new token
            new_token = self.decoder[pair[0][0]] + self.decoder[pair[0][1]]
            self.encoder[new_token] = len(self.encoder)
            self.decoder[len(self.encoder) - 1] = new_token
            
            # Merge pairs in the data
            for word_idx, word in enumerate(word_ids):
                i = 0
                while i < len(word) - 1:
                    if (word[i], word[i + 1]) == pair[0]:
                        word[i:i + 2] = [self.encoder[new_token]]
                    else:
                        i += 1
        
        print(f"Vocabulary size: {len(self.encoder)}")
    
    def save(self, path: str):
        """Save tokenizer to file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'encoder': self.encoder,
                'decoder': {str(k): v for k, v in self.decoder.items()},
                'vocab_size': self.vocab_size,
                'special_tokens': self.special_tokens
            }, f, ensure_ascii=False, indent=2)
    
    def load(self, path: str):
        """Load tokenizer from file"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.encoder = data['encoder']
            self.decoder = {int(k): v for k, v in data['decoder'].items()}
            self.vocab_size = data['vocab_size']
            self.special_tokens = data['special_tokens']
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token ids"""
        if not text:
            return []
            
        tokens = []
        for word in text.split():
            # Try to encode whole word first
            if word in self.encoder:
                tokens.append(self.encoder[word])
                continue
            
            # Otherwise encode character by character
            for char in word:
                tokens.append(self.encoder.get(char, self.encoder['<unk>']))
            
            # Add space between words (if not last word)
            if word != text.split()[-1]:
                tokens.append(self.encoder.get(' ', self.encoder['<unk>']))
        
        return tokens
    
    def decode(self, ids: List[int]) -> str:
        """Decode token ids to text"""
        return ''.join(self.decoder.get(id, '<unk>') for id in ids) 