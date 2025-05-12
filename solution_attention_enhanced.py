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
    
    def forward(self, x):
        """
        Apply enhanced self-attention to the input sequence.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, embedding_dim)
            
        Returns:
            Output tensor of shape (batch_size, sequence_length, embedding_dim)
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
        
        return output

class SimpleSelfAttention(nn.Module):
    """
    Simple self-attention layer as described in the assignment.
    """
    def __init__(self, embedding_dim):
        super(SimpleSelfAttention, self).__init__()
        
        # Scaling factor
        self.scale = math.sqrt(embedding_dim)
    
    def forward(self, x):
        """
        Apply simple self-attention to the input sequence.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, embedding_dim)
            
        Returns:
            Output tensor of shape (batch_size, sequence_length, embedding_dim)
        """
        # Compute scaled attention weights
        attention_weights = torch.bmm(x, x.transpose(1, 2)) / self.scale
        
        # Apply softmax along the last dimension
        attention_weights = F.softmax(attention_weights, dim=2)
        
        # Compute the weighted sum
        output = torch.bmm(attention_weights, x)
        
        return output

class AttentionPoolingModel(nn.Module):
    """
    Model with embedding, self-attention, global pooling, and linear projection.
    """
    def __init__(self, vocab_size, embedding_dim, num_classes, pooling_type='max', attention_type='simple'):
        super(AttentionPoolingModel, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Self-attention layer
        if attention_type == 'simple':
            self.attention = SimpleSelfAttention(embedding_dim)
        elif attention_type == 'enhanced':
            self.attention = EnhancedSelfAttention(embedding_dim)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        # Linear projection
        self.linear = nn.Linear(embedding_dim, num_classes)
        
        # Store pooling type
        self.pooling_type = pooling_type
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Embed the input: (batch_size, sequence_length, embedding_dim)
        embedded = self.embedding(x)
        
        # Apply self-attention: (batch_size, sequence_length, embedding_dim)
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
        
        return output

class PoolingModel(nn.Module):
    """
    Original model without self-attention (for comparison).
    """
    def __init__(self, vocab_size, embedding_dim, num_classes, pooling_type='max'):
        super(PoolingModel, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Linear projection
        self.linear = nn.Linear(embedding_dim, num_classes)
        
        # Store pooling type
        self.pooling_type = pooling_type
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Embed the input
        embedded = self.embedding(x)  # (batch_size, sequence_length, embedding_dim)
        
        # Apply pooling
        if self.pooling_type == 'max':
            # Global max pooling
            pooled, _ = torch.max(embedded, dim=1)  # (batch_size, embedding_dim)
        elif self.pooling_type == 'avg':
            # Global average pooling
            pooled = torch.mean(embedded, dim=1)  # (batch_size, embedding_dim)
        elif self.pooling_type == 'sum':
            # Global sum pooling
            pooled = torch.sum(embedded, dim=1)  # (batch_size, embedding_dim)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")
        
        # Project down to the number of classes
        output = self.linear(pooled)  # (batch_size, num_classes)
        
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

def compare_models(x_train, y_train, x_val, y_val, w2i, base_batch_size=32, subset_size=5000):
    """
    Compare models with and without self-attention
    
    Args:
        x_train: Training sequences
        y_train: Training labels
        x_val: Validation sequences
        y_val: Validation labels
        w2i: Word to index mapping
        base_batch_size: Base batch size
        subset_size: Size of the subset to use for training
    """
    # Vocabulary size
    vocab_size = len(w2i)
    num_classes = 2
    
    # Use a subset of the data for faster training
    subset_size = min(subset_size, len(x_train))
    x_train_subset = x_train[:subset_size]
    y_train_subset = y_train[:subset_size]
    
    # Create batches
    train_batches = imdb_loader.create_batches(x_train_subset, y_train_subset, base_batch_size, w2i)
    val_batches = imdb_loader.create_batches(x_val, y_val, base_batch_size, w2i)
    
    # Fixed hyperparameters
    embedding_dim = 100
    pooling_type = 'max'
    num_epochs = 10  # Increased from 5 to 10
    
    # Train model without self-attention
    print("\n=== Training Model without Self-Attention ===")
    model_without_attention = PoolingModel(vocab_size, embedding_dim, num_classes, pooling_type=pooling_type)
    _, (losses_without, accuracies_without) = train_model(
        model_without_attention, train_batches, val_batches, num_epochs=num_epochs
    )
    
    # Train model with simple self-attention
    print("\n=== Training Model with Simple Self-Attention ===")
    model_with_simple_attention = AttentionPoolingModel(
        vocab_size, embedding_dim, num_classes, 
        pooling_type=pooling_type, attention_type='simple'
    )
    _, (losses_simple, accuracies_simple) = train_model(
        model_with_simple_attention, train_batches, val_batches, num_epochs=num_epochs
    )
    
    # Train model with enhanced self-attention
    print("\n=== Training Model with Enhanced Self-Attention ===")
    model_with_enhanced_attention = AttentionPoolingModel(
        vocab_size, embedding_dim, num_classes, 
        pooling_type=pooling_type, attention_type='enhanced'
    )
    _, (losses_enhanced, accuracies_enhanced) = train_model(
        model_with_enhanced_attention, train_batches, val_batches, num_epochs=num_epochs
    )
    
    # Plot comparison
    plt.figure(figsize=(12, 5))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(losses_without, label='Without Self-Attention')
    plt.plot(losses_simple, label='Simple Self-Attention')
    plt.plot(losses_enhanced, label='Enhanced Self-Attention')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(accuracies_without, label='Without Self-Attention')
    plt.plot(accuracies_simple, label='Simple Self-Attention')
    plt.plot(accuracies_enhanced, label='Enhanced Self-Attention')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('attention_comparison_enhanced.png')
    
    # Print final results
    print("\n=== Final Results ===")
    print("Model                   | Final Loss | Final Accuracy")
    print("------------------------|------------|---------------")
    print(f"Without Self-Attention    | {losses_without[-1]:10.4f} | {accuracies_without[-1]:14.2f}%")
    print(f"Simple Self-Attention     | {losses_simple[-1]:10.4f} | {accuracies_simple[-1]:14.2f}%")
    print(f"Enhanced Self-Attention   | {losses_enhanced[-1]:10.4f} | {accuracies_enhanced[-1]:14.2f}%")

def main():
    # Load the IMDb dataset
    (x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = imdb_loader.load_imdb_data(final=False)
    
    # Print dataset information
    print(f'Training examples: {len(x_train)}, Validation examples: {len(x_val)}')
    print(f'Vocabulary size: {len(i2w)}, Number of classes: {numcls}')
    print(f'Example review: {[i2w[w] for w in x_train[0][:20]]}...')
    
    # Compare models with and without self-attention
    compare_models(x_train, y_train, x_val, y_val, w2i)
    
    print("\nTraining completed!")
    print("Check attention_comparison_enhanced.png for visualization.")

if __name__ == "__main__":
    main()
