import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import matplotlib.pyplot as plt
import time
import imdb_loader

# Set random seed for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

class PoolingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, pooling_type='max', embedding_init=None):
        """
        Model with an embedding layer, a global pooling operation, and a linear projection
        
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of the embedding vectors
            num_classes: Number of output classes
            pooling_type: Type of pooling ('max', 'avg', or 'sum')
            embedding_init: Optional initialization for the embedding layer
        """
        super(PoolingModel, self).__init__()
        
        # Create embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Initialize embedding if provided
        if embedding_init is not None:
            self.embedding.weight.data.copy_(embedding_init)
        
        # Linear projection layer
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

def main():
    # Load the IMDb dataset
    (x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = imdb_loader.load_imdb_data(final=False)
    
    # Print dataset information
    print(f'Training examples: {len(x_train)}, Validation examples: {len(x_val)}')
    print(f'Vocabulary size: {len(i2w)}, Number of classes: {numcls}')
    print(f'Example review: {[i2w[w] for w in x_train[0][:20]]}...')
    
    # Create batches with variable batch sizes
    base_batch_size = 64
    train_batches = imdb_loader.create_batches(x_train, y_train, base_batch_size, w2i, variable_batch_size=True)
    val_batches = imdb_loader.create_batches(x_val, y_val, base_batch_size, w2i, variable_batch_size=True)
    
    # Model parameters
    vocab_size = len(i2w)
    embedding_dim = 300
    
    # Train models with different pooling types
    pooling_types = ['max', 'avg', 'sum']
    results = []
    
    for pooling_type in pooling_types:
        print(f"\n=== Training Model with {pooling_type} Pooling ===")
        model = PoolingModel(vocab_size, embedding_dim, numcls, pooling_type=pooling_type)
        _, (train_losses, val_accuracies) = train_model(model, train_batches, val_batches, num_epochs=10)
        
        results.append({
            'pooling_type': pooling_type,
            'final_loss': train_losses[-1],
            'final_accuracy': val_accuracies[-1],
            'accuracies': val_accuracies
        })
    
    # Print results
    print("\n=== Final Results ===")
    print("Pooling Type | Final Loss | Final Accuracy")
    print("------------|------------|---------------")
    
    for result in sorted(results, key=lambda x: x['final_accuracy'], reverse=True):
        print(f"{result['pooling_type']:11s} | {result['final_loss']:10.4f} | {result['final_accuracy']:14.2f}%")
    
    # Plot accuracy over epochs for each pooling type
    plt.figure(figsize=(12, 6))
    
    for result in results:
        plt.plot(result['accuracies'], label=f"{result['pooling_type']} pooling")
    
    plt.title('Validation Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('pooling_comparison_full.png')
    
    print("\nTraining completed!")
    print("Check pooling_comparison_full.png for visualization.")

if __name__ == "__main__":
    main()
