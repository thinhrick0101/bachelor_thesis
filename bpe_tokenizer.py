import os
import json
import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm  # For progress tracking
from scipy import stats

class BPETokenizer:
    """
    An advanced implementation of Byte-Pair Encoding (BPE) tokenizer
    with sophisticated training process and subword regularization
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
        self.max_token_length = 0
        self.char_ngram_freqs = defaultdict(int)  # Track character n-gram frequencies
        self.token_coverage = defaultdict(int)    # Track token coverage
        
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
        
        # Initialize n-gram frequencies (up to trigrams)
        print("Analyzing character n-grams...")
        for i in range(len(text) - 2):
            self.char_ngram_freqs[text[i]] += 1
            self.char_ngram_freqs[text[i:i+2]] += 1
            self.char_ngram_freqs[text[i:i+3]] += 1
        
        self.max_token_length = 1
        print(f"Initial vocabulary size: {len(self.encoder)} tokens")
        print(f"Character n-gram types: {len(self.char_ngram_freqs):,}")
    
    def _get_stats(self, ids: List[List[int]], min_frequency: int) -> Tuple[Counter, Dict[Tuple[int, int], float]]:
        """Get counts of adjacent pairs with sophisticated scoring"""
        pairs = Counter()
        pair_scores = {}
        
        # Process each piece
        for piece in ids:
            if len(piece) < 2:
                continue
            
            # Count pairs in current piece
            for i in range(len(piece) - 1):
                pair = (piece[i], piece[i + 1])
                pairs[pair] += 1
                
                # Calculate pair score based on multiple factors
                if pairs[pair] >= min_frequency:
                    token_a = self.decoder[pair[0]]
                    token_b = self.decoder[pair[1]]
                    merged = token_a + token_b
                    
                    # Factors for scoring:
                    # 1. Raw frequency
                    freq_score = pairs[pair] / sum(pairs.values())
                    
                    # 2. Character n-gram probability
                    ngram_prob = self.char_ngram_freqs.get(merged, 0) / sum(self.char_ngram_freqs.values())
                    
                    # 3. Length penalty (prefer shorter tokens early, longer tokens later)
                    vocab_progress = len(self.encoder) / self.vocab_size
                    length_penalty = 1.0 if vocab_progress < 0.5 else np.log(len(merged) + 1)
                    
                    # 4. Coverage improvement potential
                    coverage_score = 1.0
                    if merged in self.token_coverage:
                        coverage_score = np.log(self.token_coverage[merged] + 1)
                    
                    # Combine scores
                    final_score = (freq_score * 0.4 + 
                                 ngram_prob * 0.3 + 
                                 length_penalty * 0.2 + 
                                 coverage_score * 0.1)
                    
                    pair_scores[pair] = final_score
        
        return pairs, pair_scores
    
    def _check_convergence(self, counts: List[int], window_size: int = 50) -> bool:
        """Advanced convergence detection using statistical analysis"""
        if len(counts) < window_size:
            return False
            
        recent_counts = counts[-window_size:]
        
        # 1. Mann-Kendall trend test
        trend, p_value = stats.kendalltau(range(window_size), recent_counts)
        
        # 2. Check for plateau using rolling statistics
        mean_diff = np.mean(np.diff(recent_counts))
        std_diff = np.std(np.diff(recent_counts))
        cv = std_diff / abs(mean_diff) if mean_diff != 0 else float('inf')
        
        # 3. Check vocabulary coverage
        vocab_ratio = len(self.encoder) / self.vocab_size
        
        # Combined convergence criteria
        return (
            (p_value > 0.05 or abs(trend) < 0.1) and  # No significant trend
            cv > 10.0 and                             # High variation relative to change
            vocab_ratio > 0.95                        # Close to target vocabulary size
        )
    
    def _update_token_coverage(self, text: str, batch_size: int = 1000000):
        """Update token coverage statistics"""
        for i in range(0, len(text), batch_size):
            batch = text[i:i + batch_size]
            for j in range(len(batch)):
                for length in range(1, min(len(batch) - j, self.max_token_length + 1)):
                    subword = batch[j:j + length]
                    self.token_coverage[subword] += 1
    
    def train(self, text: str, min_frequency: int = 1, batch_size: int = 1000000):
        """
        Train the tokenizer with advanced techniques and adaptive processing
        """
        try:
            # Initialize vocabulary and analyze n-grams
            self._init_vocab(text)
            
            # Initial token coverage analysis
            print("Analyzing token coverage...")
            self._update_token_coverage(text[:min(len(text), 10000000)])  # Sample first 10M chars
            
            # Convert text to base tokens in batches with adaptive size
            print("Converting text to base tokens...")
            words = []
            total_chars = len(text)
            
            # Adjust batch size based on average memory usage
            chars_per_mb = 100000  # Approximate chars that consume 1MB
            available_memory = 1000  # Target memory usage in MB
            adaptive_batch_size = min(batch_size, 
                                    chars_per_mb * available_memory)
            
            total_batches = (total_chars + adaptive_batch_size - 1) // adaptive_batch_size
            for i in range(0, total_chars, adaptive_batch_size):
                print(f"Processing batch {i//adaptive_batch_size + 1}/{total_batches}")
                batch = text[i:i + adaptive_batch_size]
                words.extend([[c for c in word] for word in batch.split()])
            
            print("Converting words to token IDs...")
            word_ids = [[self.encoder[c] for c in word] for word in words]
            print(f"Initial token count: {sum(len(w) for w in word_ids):,}")
            
            # Training progress tracking
            total_tokens = self.vocab_size - len(self.encoder)
            print(f"Training BPE tokenizer to add {total_tokens:,} new tokens")
            pbar = tqdm(total=total_tokens, desc="Training BPE tokenizer")
            
            # Initialize tracking variables
            pair_counts = []
            freq_distribution = {}
            iteration = 0
            
            while len(self.encoder) < self.vocab_size:
                iteration += 1
                
                # Get pair statistics and scores
                pairs, pair_scores = self._get_stats(word_ids, min_frequency)
                if not pairs:
                    print("\nNo more pairs found above minimum frequency")
                    break
                
                # Select best pair based on scores
                if pair_scores:
                    pair = max(pair_scores.items(), key=lambda x: x[1])[0]
                else:
                    pair = max(pairs.items(), key=lambda x: x[1])[0]
                
                current_count = pairs[pair]
                pair_counts.append(current_count)
                
                # Update frequency distribution
                freq_bucket = current_count // 10 * 10
                freq_distribution[freq_bucket] = freq_distribution.get(freq_bucket, 0) + 1
                
                # Detailed progress logging
                if iteration % 100 == 0 or len(self.encoder) % 1000 == 0:
                    print(f"\nIteration {iteration}")
                    print(f"Vocabulary size: {len(self.encoder):,}")
                    print(f"Most frequent pair: {self.decoder[pair[0]]}{self.decoder[pair[1]]} "
                          f"(count: {current_count:,})")
                    if pair in pair_scores:
                        print(f"Pair score: {pair_scores[pair]:.4f}")
                    print(f"Maximum token length: {self.max_token_length}")
                
                # Check for convergence
                if self._check_convergence(pair_counts):
                    print("\nConverged based on statistical analysis")
                    break
                
                # Create new token
                new_token = self.decoder[pair[0]] + self.decoder[pair[1]]
                self.encoder[new_token] = len(self.encoder)
                self.decoder[len(self.encoder) - 1] = new_token
                
                # Update maximum token length and coverage
                self.max_token_length = max(self.max_token_length, len(new_token))
                self.token_coverage[new_token] = pairs[pair]
                
                # Merge pairs in the data
                merged_count = 0
                for word_idx, word in enumerate(word_ids):
                    i = 0
                    while i < len(word) - 1:
                        if (word[i], word[i + 1]) == pair:
                            word[i:i + 2] = [self.encoder[new_token]]
                            merged_count += 1
                        else:
                            i += 1
                
                pbar.update(1)
                if iteration % 100 == 0:
                    print(f"Merged {merged_count:,} pairs in this iteration")
            
            pbar.close()
            
            # Final statistics
            print(f"\nFinal vocabulary size: {len(self.encoder):,}")
            print(f"Maximum token length: {self.max_token_length}")
            print(f"Reached {len(self.encoder)/self.vocab_size*100:.1f}% of target vocabulary size")
            
            # Print frequency distribution
            print("\nFinal frequency distribution:")
            for freq in sorted(freq_distribution.keys(), reverse=True):
                print(f"  {freq}-{freq+10}: {freq_distribution[freq]} tokens")
            
            # Print coverage statistics
            total_subwords = sum(self.token_coverage.values())
            print("\nToken coverage statistics:")
            print(f"Total subword occurrences: {total_subwords:,}")
            print(f"Average occurrences per token: {total_subwords/len(self.encoder):,.1f}")
            
        except Exception as e:
            print(f"\nError during tokenizer training: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
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