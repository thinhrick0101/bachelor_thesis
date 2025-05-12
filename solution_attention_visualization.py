import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import matplotlib.pyplot as plt
import time
import imdb_loader
import math
import seaborn as sns

# Set random seed for reproducibility
torch.manual_seed(42)
random.seed(42)
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

def train_model(model, train_batches, val_batches, num_epochs=5, learning_rate=0.001):
    """
    Train the model and evaluate on validation set
    
    Args:
        model: The model to train
        train_batches: List of training batches
        val_batches: List of validation batches
        num_epochs: Number of training epochs
        learning_rate: Learning rate for the optimizer
        
    Returns:
        Trained model and training metrics
    """
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # For tracking metrics
    train_losses = []
    val_accuracies = []
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        start_time = time.time()
        
        # Process each batch
        for inputs, targets in train_batches:
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Calculate average loss for the epoch
        avg_loss = total_loss / num_batches
        train_losses.append(avg_loss)
        
        # Evaluate on validation set
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_batches:
                # Forward pass
                outputs = model(inputs)
                
                # Get predictions
                _, predicted = torch.max(outputs.data, 1)
                
                # Update statistics
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        # Calculate accuracy
        accuracy = 100 * correct / total
        val_accuracies.append(accuracy)
        
        # Calculate epoch time
        epoch_time = time.time() - start_time
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.2f}%, Time: {epoch_time:.2f}s')
    
    return model, (train_losses, val_accuracies)

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
    avg_attentions = []
    for i, word in enumerate(words):
        # Get the average attention this word receives
        avg_attention = attention_weights[:, i].mean()
        avg_attentions.append((word, avg_attention))
    
    # Sort by average attention
    avg_attentions.sort(key=lambda x: x[1], reverse=True)
    
    # Print top 10 most attended words
    print("\nTop 10 most attended words:")
    for word, avg_attention in avg_attentions[:10]:
        print(f"Word: {word}, Average attention: {avg_attention:.4f}")

def find_examples_with_negation(x_val, y_val, i2w, num_examples=5, max_length=50):
    """
    Find examples with negation words
    
    Args:
        x_val: Validation sequences
        y_val: Validation labels
        i2w: Index to word mapping
        num_examples: Number of examples to find
        max_length: Maximum sequence length
        
    Returns:
        List of (index, words, label) tuples
    """
    negation_words = ['not', 'no', 'never', "don't", "doesn't", "isn't", "wasn't", "aren't", "weren't", "haven't", "hasn't", "hadn't", "won't", "wouldn't", "couldn't", "shouldn't"]
    
    examples = []
    for i, seq in enumerate(x_val):
        if len(seq) > max_length:
            continue
            
        words = [i2w[idx].lower() for idx in seq]
        
        # Check if any negation word is in the sequence
        if any(neg_word in words for neg_word in negation_words):
            examples.append((i, words, y_val[i]))
            
            if len(examples) >= num_examples:
                break
    
    return examples

def main():
    # Load the IMDb dataset
    (x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = imdb_loader.load_imdb_data(final=False)
    
    # Print dataset information
    print(f'Training examples: {len(x_train)}, Validation examples: {len(x_val)}')
    print(f'Vocabulary size: {len(i2w)}, Number of classes: {numcls}')
    
    # Use a subset of the data for faster training
    subset_size = 5000
    x_train_subset = x_train[:subset_size]
    y_train_subset = y_train[:subset_size]
    
    # Create batches
    base_batch_size = 32
    train_batches = imdb_loader.create_batches(x_train_subset, y_train_subset, base_batch_size, w2i)
    val_batches = imdb_loader.create_batches(x_val, y_val, base_batch_size, w2i)
    
    # Create model
    vocab_size = len(i2w)
    embedding_dim = 100
    model = AttentionPoolingModel(vocab_size, embedding_dim, numcls, pooling_type='avg')
    
    # Train model
    print("\n=== Training Model with Self-Attention ===")
    trained_model, _ = train_model(model, train_batches, val_batches, num_epochs=5)
    
    # Find examples with negation
    print("\n=== Finding Examples with Negation ===")
    examples = find_examples_with_negation(x_val, y_val, i2w)
    
    if examples:
        # Print found examples
        print("\nExamples with negation:")
        for i, (idx, words, label) in enumerate(examples):
            print(f"{i+1}. {' '.join(words[:20])}... (Label: {'Positive' if label == 0 else 'Negative'})")
        
        # Choose an example
        example_idx = examples[0][0]
        
        # Visualize attention
        print("\n=== Visualizing Attention ===")
        visualize_attention(trained_model, x_val, y_val, i2w, example_idx=example_idx)
    else:
        # If no examples with negation are found, just use the first example
        visualize_attention(trained_model, x_val, y_val, i2w, example_idx=0)
    
    print("\nAttention visualization saved to attention_visualization.png")

if __name__ == "__main__":
    main()
