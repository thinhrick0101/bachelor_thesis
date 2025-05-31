"""
Dataset and tokenizer implementations for byte-level language modeling.
"""

import torch
import numpy as np
from torch.utils.data import Dataset


class ByteTokenizer:
    """
    Simple byte-level tokenizer that maps each byte to an integer in [0, 255].
    """
    def encode(self, text):
        """Convert text string or bytes to list of byte values."""
        if isinstance(text, str):
            return list(text.encode('utf-8'))
        return list(text)
    
    def decode(self, tokens):
        """Convert list of byte values back to text string."""
        return bytes(tokens).decode('utf-8', errors='replace')


class ByteDataset(Dataset):
    """
    Dataset for byte-level language modeling that loads data from a binary file.
    
    The dataset creates overlapping sequences of the specified length,
    where each sequence is used to predict the next byte in the sequence.
    """
    
    def __init__(self, path, seq_length, stride=None):
        """
        Initialize the dataset.
        
        Args:
            path: Path to the binary data file
            seq_length: Length of sequences to generate
            stride: Stride between sequences (defaults to seq_length)
        """
        # Load the entire file into memory
        with open(path, 'rb') as f:
            data = f.read()
        self.data = np.frombuffer(data, dtype=np.uint8)
        
        self.seq_length = seq_length
        self.stride = stride if stride is not None else seq_length
        
        # Calculate number of sequences
        self.n_sequences = max(0, (len(self.data) - seq_length) // self.stride)
    
    def __len__(self):
        return self.n_sequences
    
    def __getitem__(self, idx):
        """
        Get a sequence and its target.
        
        Returns:
            data: Sequence of bytes [seq_length]
            target: Same sequence shifted by 1 [seq_length]
        """
        # Get start position for this sequence
        start_idx = idx * self.stride
        
        # Get sequence and target
        sequence = self.data[start_idx:start_idx + self.seq_length]
        target = self.data[start_idx:start_idx + self.seq_length]
        
        # Convert to tensors
        sequence = torch.from_numpy(sequence.astype(np.int64))
        target = torch.from_numpy(target.astype(np.int64))
        
        return sequence, target


def create_dataloaders(
    train_path,
    val_path,
    seq_length,
    batch_size,
    num_workers=4,
    train_stride=None,
    val_stride=None
):
    """
    Create train and validation dataloaders.
    
    Args:
        train_path: Path to training data
        val_path: Path to validation data
        seq_length: Length of sequences
        batch_size: Batch size
        num_workers: Number of workers for data loading
        train_stride: Stride for training sequences (default: seq_length)
        val_stride: Stride for validation sequences (default: seq_length)
    
    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
    """
    from torch.utils.data import DataLoader
    
    # Create datasets
    train_dataset = ByteDataset(
        path=train_path,
        seq_length=seq_length,
        stride=train_stride
    )
    
    val_dataset = ByteDataset(
        path=val_path,
        seq_length=seq_length,
        stride=val_stride
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader 