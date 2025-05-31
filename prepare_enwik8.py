"""
Prepare the Enwik8 dataset by splitting it into train, validation, and test sets.
Following standard practice for Enwik8:
- First 90M bytes for training
- Next 5M bytes for validation
- Final 5M bytes for testing
"""

import os
import urllib.request
import gzip
from pathlib import Path
import numpy as np

def download_enwik8(data_dir='data/enwik8'):
    """Download and prepare the enwik8 dataset."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # URLs for the dataset
    url = 'https://data.deepai.org/enwik8.zip'
    
    # Download and extract
    print("Downloading enwik8 dataset...")
    raw_path = data_dir / 'enwik8.gz'
    if not raw_path.exists():
        urllib.request.urlretrieve(url, raw_path)
    
    # Extract and split into train/val/test
    if not (data_dir / 'train.txt').exists():
        print("Extracting and splitting dataset...")
        with gzip.open(raw_path, 'rb') as f:
            text = f.read()
        
        # Split sizes (90M/5M/5M)
        train_size = 90_000_000
        val_size = 5_000_000
        
        # Save splits
        with open(data_dir / 'train.txt', 'wb') as f:
            f.write(text[:train_size])
        
        with open(data_dir / 'val.txt', 'wb') as f:
            f.write(text[train_size:train_size + val_size])
        
        with open(data_dir / 'test.txt', 'wb') as f:
            f.write(text[train_size + val_size:])
        
        print("Dataset prepared successfully!")
        print(f"Train size: {train_size/1_000_000:.1f}M")
        print(f"Val size: {val_size/1_000_000:.1f}M")
        print(f"Test size: {val_size/1_000_000:.1f}M")

def split_enwik8(data_path, output_dir):
    """
    Split the Enwik8 dataset into train, validation, and test sets.
    
    Args:
        data_path: Path to the original enwik8 file
        output_dir: Directory to save the split files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the entire file
    with open(data_path, 'rb') as f:
        data = f.read()
    
    # Split sizes (in bytes)
    train_size = 90_000_000  # 90M
    val_size = 5_000_000    # 5M
    test_size = 5_000_000   # 5M
    
    # Verify we have enough data
    total_size = train_size + val_size + test_size
    assert len(data) >= total_size, f"Enwik8 file too small: {len(data)} bytes < {total_size} bytes"
    
    # Split the data
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:train_size + val_size + test_size]
    
    # Save splits
    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    
    for name, split_data in splits.items():
        output_path = os.path.join(output_dir, f'{name}.bin')
        with open(output_path, 'wb') as f:
            f.write(split_data)
        print(f"Saved {name} split ({len(split_data):,} bytes) to {output_path}")

def main():
    # Paths
    data_path = os.path.join("bachelor_thesis", "data", "enwik8")
    output_dir = os.path.join("bachelor_thesis", "data", "enwik8_splits")
    
    # Split the dataset
    split_enwik8(data_path, output_dir)
    print("\nDataset preparation complete!")

if __name__ == '__main__':
    main() 