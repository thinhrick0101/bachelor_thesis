import os
import urllib.request
import gzip
from pathlib import Path

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

if __name__ == '__main__':
    download_enwik8() 