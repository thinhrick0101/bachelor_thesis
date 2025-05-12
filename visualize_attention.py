import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import imdb_loader
import math
import seaborn as sns

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class EnhancedSelfAttention(nn.Module):
    """
    Enhanced self-attention layer with scaling and projections.
    """
    def __init__(self, embedding_dim):
        super(EnhancedSelfAttention, self).__init__()
        
        # Projections for query, key, and value
        self.query_proj = nn.Linear(embedding_dim, embedding_dim)
        self.key_proj = nn.Linear(embedding_dim, embedding_dim)
        self.value_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # Scaling factor
        self.scale = math.sqrt(embedding_dim)
    
    def forward(self, x, return_attention=False):
        """
        Apply enhanced self-attention to the input sequence.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, embedding_dim)
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor of shape (batch_size, sequence_length, embedding_dim)
            Attention weights if return_attention=True
        """
        # Project input to query, key, and value
        query = self.query_proj(x)
        key = self.key_proj(x)
        value = self.value_proj(x)
        
        # Compute scaled dot-product attention
        # (batch_size, sequence_length, sequence_length)
        attention_scores = torch.bmm(query, key.transpose(1, 2)) / self.scale
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=2)
        
        # Compute weighted sum
        output = torch.bmm(attention_weights, value)
        
        if return_attention:
            return output, attention_weights
        else:
            return output

class AttentionPoolingModel(nn.Module):
    """
    Model with embedding, self-attention, global pooling, and linear projection.
    """
    def __init__(self, vocab_size, embedding_dim, num_classes, pooling_type='max'):
        super(AttentionPoolingModel, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Self-attention layer
        self.attention = EnhancedSelfAttention(embedding_dim)
        
        # Linear projection
        self.linear = nn.Linear(embedding_dim, num_classes)
        
        # Store pooling type
        self.pooling_type = pooling_type
    
    def forward(self, x, return_attention=False):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
            Attention weights if return_attention=True
        """
        # Embed the input: (batch_size, sequence_length, embedding_dim)
        embedded = self.embedding(x)
        
        # Apply self-attention: (batch_size, sequence_length, embedding_dim)
        if return_attention:
            attended, attention_weights = self.attention(embedded, return_attention=True)
        else:
            attended = self.attention(embedded)
        
        # Apply pooling
        if self.pooling_type == 'max':
            # Global max pooling
            pooled, _ = torch.max(attended, dim=1)  # (batch_size, embedding_dim)
        elif self.pooling_type == 'avg':
            # Global average pooling
            pooled = torch.mean(attended, dim=1)  # (batch_size, embedding_dim)
        elif self.pooling_type == 'sum':
            # Global sum pooling
            pooled = torch.sum(attended, dim=1)  # (batch_size, embedding_dim)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")
        
        # Project down to the number of classes
        output = self.linear(pooled)  # (batch_size, num_classes)
        
        if return_attention:
            return output, attention_weights
        else:
            return output

def visualize_attention(model, x_val, y_val, i2w, example_idx=0, max_length=50):
    """
    Visualize attention weights for a specific example
    
    Args:
        model: Trained model
        x_val: Validation sequences
        y_val: Validation labels
        i2w: Index to word mapping
        example_idx: Index of the example to visualize
        max_length: Maximum sequence length to visualize
    """
    # Set model to evaluation mode
    model.eval()
    
    # Get example
    example = x_val[example_idx]
    label = y_val[example_idx]
    
    # Truncate if too long
    if len(example) > max_length:
        example = example[:max_length]
    
    # Convert to tensor
    example_tensor = torch.tensor([example], dtype=torch.long)
    
    # Get attention weights
    with torch.no_grad():
        _, attention_weights = model(example_tensor, return_attention=True)
    
    # Convert to numpy
    attention_weights = attention_weights[0].numpy()
    
    # Get words
    words = [i2w[idx] for idx in example]
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot attention weights
    sns.heatmap(
        attention_weights[:len(words), :len(words)],
        xticklabels=words,
        yticklabels=words,
        cmap='viridis',
        annot=False
    )
    
    plt.title(f'Attention Weights (Label: {"Positive" if label == 0 else "Negative"})')
    plt.xlabel('Key')
    plt.ylabel('Query')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('attention_visualization.png')
    
    # Print the example text
    print("Example text:")
    print(" ".join(words))
    print(f"Label: {'Positive' if label == 0 else 'Negative'}")
    
    # Find the most attended words
    for i, word in enumerate(words):
        # Get the average attention this word receives
        avg_attention = attention_weights[:, i].mean()
        print(f"Word: {word}, Average attention: {avg_attention:.4f}")

def main():
    # Load the IMDb dataset
    (x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = imdb_loader.load_imdb_data(final=False)
    
    # Print dataset information
    print(f'Training examples: {len(x_train)}, Validation examples: {len(x_val)}')
    print(f'Vocabulary size: {len(i2w)}, Number of classes: {numcls}')
    
    # Create a model
    vocab_size = len(i2w)
    embedding_dim = 100
    model = AttentionPoolingModel(vocab_size, embedding_dim, numcls, pooling_type='avg')
    
    # Find examples with specific phrases
    examples_with_not = []
    for i, seq in enumerate(x_val):
        words = [i2w[idx] for idx in seq]
        if 'not' in words and len(seq) < 50:  # Find examples with "not" that aren't too long
            examples_with_not.append((i, words))
    
    if examples_with_not:
        # Print found examples
        print("\nExamples with 'not':")
        for i, (idx, words) in enumerate(examples_with_not[:5]):
            print(f"{i+1}. {' '.join(words[:20])}... (Label: {'Positive' if y_val[idx] == 0 else 'Negative'})")
        
        # Choose an example
        example_idx = examples_with_not[0][0]
        
        # Visualize attention
        visualize_attention(model, x_val, y_val, i2w, example_idx=example_idx)
    else:
        # If no examples with "not" are found, just use the first example
        visualize_attention(model, x_val, y_val, i2w, example_idx=0)
    
    print("\nAttention visualization saved to attention_visualization.png")

if __name__ == "__main__":
    main()
