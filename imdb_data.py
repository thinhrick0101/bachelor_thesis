import torch
import random
import numpy as np

# Create a small synthetic dataset for testing
def create_synthetic_imdb():
    # Define a small vocabulary
    words = ["good", "bad", "excellent", "terrible", "great", "awful", "amazing", "horrible", 
             "film", "movie", "acting", "plot", "story", "character", "director", "scene", 
             "the", "a", "an", "and", "but", "or", "in", "of", "to", "with", "was", "is", "are", "were"]
    
    # Create word to index and index to word mappings
    i2w = [".pad", ".start", ".end", ".unk"] + words
    w2i = {t:i for i, t in enumerate(i2w)}
    
    # Generate synthetic reviews
    positive_templates = [
        ["the", "movie", "was", "good"],
        ["excellent", "film", "with", "great", "acting"],
        ["amazing", "plot", "and", "character"],
        ["the", "director", "is", "great"]
    ]
    
    negative_templates = [
        ["the", "movie", "was", "bad"],
        ["terrible", "film", "with", "awful", "acting"],
        ["horrible", "plot", "and", "character"],
        ["the", "director", "is", "awful"]
    ]
    
    # Create training data
    x_train, y_train = [], []
    
    # Generate 100 positive examples
    for _ in range(100):
        template = random.choice(positive_templates)
        # Add some random words to make reviews of different lengths
        review = template + random.sample(words, random.randint(0, 10))
        x_train.append([w2i[w] for w in review])
        y_train.append(0)  # 0 for positive
    
    # Generate 100 negative examples
    for _ in range(100):
        template = random.choice(negative_templates)
        # Add some random words to make reviews of different lengths
        review = template + random.sample(words, random.randint(0, 10))
        x_train.append([w2i[w] for w in review])
        y_train.append(1)  # 1 for negative
    
    # Create validation data
    x_val, y_val = [], []
    
    # Generate 20 positive examples
    for _ in range(20):
        template = random.choice(positive_templates)
        # Add some random words to make reviews of different lengths
        review = template + random.sample(words, random.randint(0, 10))
        x_val.append([w2i[w] for w in review])
        y_val.append(0)  # 0 for positive
    
    # Generate 20 negative examples
    for _ in range(20):
        template = random.choice(negative_templates)
        # Add some random words to make reviews of different lengths
        review = template + random.sample(words, random.randint(0, 10))
        x_val.append([w2i[w] for w in review])
        y_val.append(1)  # 1 for negative
    
    # Sort by length
    x_train, y_train = sort_by_len(x_train, y_train)
    x_val, y_val = sort_by_len(x_val, y_val)
    
    return (x_train, y_train), (x_val, y_val), (i2w, w2i), 2

def sort_by_len(data, labels):
    """
    Sort a dataset by sentence length
    """
    data, labels = zip(*sorted(zip(data, labels), key=lambda p: len(p[0])))
    return list(data), list(labels)

def load_imdb(final=False):
    """
    Load the synthetic IMDB dataset
    """
    return create_synthetic_imdb()
