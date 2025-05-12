import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import imdb_loader
import random

# Set random seed for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as described in 'Attention Is All You Need'
    """
    def __init__(self, d_model, max_seq_length=5000):
        super(SinusoidalPositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer (persistent state)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_length, embedding_dim]
            
        Returns:
            Positional encoding added to input embeddings
        """
        # Add positional encoding to input embeddings
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        return x

class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with multi-head self-attention and feed-forward network
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            src: Input tensor of shape [batch_size, seq_length, embedding_dim]
            src_mask: Mask for self-attention
            src_key_padding_mask: Mask for padding tokens
            
        Returns:
            Output tensor of shape [batch_size, seq_length, embedding_dim]
        """
        # Self-attention block (with pre-norm)
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2, attn_mask=src_mask, 
                                key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        
        # Feed-forward block (with pre-norm)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(F.gelu(self.linear1(src2))))
        src = src + self.dropout2(src2)
        
        return src

class TransformerModel(nn.Module):
    """
    Full transformer model with embedding, positional encoding, transformer layers, and classification head
    """
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, num_classes, 
                 max_seq_length=512, dropout=0.1, pad_idx=0):
        super(TransformerModel, self).__init__()
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        
        # Positional encoding
        self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_seq_length)
        
        # Special [CLS] token embedding (learnable)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Classification head
        self.classifier = nn.Linear(d_model, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters
        self._init_parameters()
        
        # Store padding index
        self.pad_idx = pad_idx
        
    def _init_parameters(self):
        """Initialize model parameters"""
        # Initialize embeddings
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        
        # Initialize classifier
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)
        
    def _create_padding_mask(self, src):
        """Create padding mask for attention"""
        # src shape: [batch_size, seq_length]
        # Create mask for padding tokens (1 for padding, 0 for non-padding)
        padding_mask = (src == self.pad_idx)
        
        # Account for the [CLS] token
        # Add a column of False at the beginning (for [CLS] token)
        cls_column = torch.zeros(src.size(0), 1, dtype=torch.bool, device=src.device)
        padding_mask = torch.cat([cls_column, padding_mask], dim=1)
        
        return padding_mask
        
    def forward(self, src):
        """
        Args:
            src: Input tensor of shape [batch_size, seq_length]
            
        Returns:
            Output tensor of shape [batch_size, num_classes]
        """
        # Create padding mask
        padding_mask = self._create_padding_mask(src)
        
        # Embed tokens
        # [batch_size, seq_length] -> [batch_size, seq_length, d_model]
        x = self.embedding(src)
        
        # Prepend [CLS] token to sequence
        # Expand cls_token to batch size
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        # Concatenate [CLS] token to beginning of each sequence
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=padding_mask)
        
        # Apply final layer normalization
        x = self.norm(x)
        
        # Extract [CLS] token representation
        # [CLS] token is at position 0
        cls_representation = x[:, 0]
        
        # Apply classification head
        output = self.classifier(cls_representation)
        
        return output

def train_model(model, train_batches, val_batches, num_epochs=5, learning_rate=0.0001, 
                weight_decay=0.01, warmup_steps=0, device=None, patience=3, label_smoothing=0.1):
    """
    Train the model and evaluate on validation set
    
    Args:
        model: The model to train
        train_batches: List of training batches
        val_batches: List of validation batches
        num_epochs: Number of training epochs
        learning_rate: Learning rate for the optimizer
        weight_decay: Weight decay for regularization
        warmup_steps: Number of warmup steps for learning rate scheduling
        device: Device to use for training (cuda or cpu)
        patience: Number of epochs to wait for improvement before early stopping
        label_smoothing: Label smoothing factor
        
    Returns:
        Trained model and training metrics
    """
    # Determine device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move model to device
    model = model.to(device)
    print(f"Training on device: {device}")
    
    # Define loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    # Define optimizer with weight decay
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    
    # Calculate total number of training steps
    total_steps = len(train_batches) * num_epochs
    
    # Create learning rate scheduler
    if warmup_steps > 0:
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    # For tracking metrics
    train_losses = []
    val_accuracies = []
    best_accuracy = 0
    best_model_state = None
    no_improvement_count = 0
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        train_correct = 0
        train_total = 0
        start_time = time.time()
        
        # Process each batch
        for inputs, targets in train_batches:
            # Move tensors to the configured device
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
            
            # Backward pass and optimize
            loss.backward()
            
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update learning rate with warmup scheduler
            if warmup_steps > 0:
                scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Calculate average loss and training accuracy for the epoch
        avg_loss = total_loss / num_batches
        train_accuracy = 100 * train_correct / train_total
        train_losses.append(avg_loss)
        
        # Evaluate on validation set
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_batches:
                # Move tensors to the configured device
                inputs = inputs.to(device)
                targets = targets.to(device)
                
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
        
        # Update learning rate with ReduceLROnPlateau scheduler
        if warmup_steps == 0:
            scheduler.step(accuracy)
        
        # Calculate epoch time
        epoch_time = time.time() - start_time
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Acc: {accuracy:.2f}%, Time: {epoch_time:.2f}s')
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_state = model.state_dict().copy()
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        # Early stopping
        if no_improvement_count >= patience:
            print(f"No improvement for {patience} epochs. Early stopping.")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation accuracy: {best_accuracy:.2f}%")
    
    return model, (train_losses, val_accuracies)

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def visualize_results(train_losses, val_accuracies):
    """
    Visualize training loss and validation accuracy
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training loss
    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Plot validation accuracy
    ax2.plot(val_accuracies)
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True)
    
    # Save figure
    plt.tight_layout()
    plt.savefig('transformer_results.png')
    plt.close()

def main():
    # Load the IMDb dataset
    (x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = imdb_loader.load_imdb_data(final=False)
    
    # Print dataset information
    print(f'Training examples: {len(x_train)}, Validation examples: {len(x_val)}')
    print(f'Vocabulary size: {len(i2w)}, Number of classes: {numcls}')
    
    # Model hyperparameters
    vocab_size = len(i2w)
    d_model = 256  # Embedding dimension
    nhead = 8  # Number of attention heads
    num_layers = 4  # Number of transformer layers
    dim_feedforward = 1024  # Dimension of feed-forward network
    dropout = 0.2  # Dropout rate
    pad_idx = w2i['.pad']  # Padding token index
    
    # Training hyperparameters
    batch_size = 32
    num_epochs = 15
    learning_rate = 1e-4
    weight_decay = 0.01
    label_smoothing = 0.1
    subset_size = 10000  # Use a subset of the data for faster training
    
    # Use a subset of the data for faster training
    subset_size = min(subset_size, len(x_train))
    x_train_subset = x_train[:subset_size]
    y_train_subset = y_train[:subset_size]
    
    # Create batches
    train_batches = imdb_loader.create_batches(x_train_subset, y_train_subset, batch_size, w2i)
    val_batches = imdb_loader.create_batches(x_val, y_val, batch_size, w2i)
    
    # Calculate warmup steps
    warmup_steps = len(train_batches) // 2  # Warmup for half an epoch
    
    # Create model
    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        num_classes=numcls,
        dropout=dropout,
        pad_idx=pad_idx
    )
    
    # Print model architecture
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {trainable_params:,} trainable out of {total_params:,} total")
    
    # Print training configuration
    print(f"\nTraining Configuration:")
    print(f"- Embedding Dimension: {d_model}")
    print(f"- Number of Attention Heads: {nhead}")
    print(f"- Number of Transformer Layers: {num_layers}")
    print(f"- Feed-forward Dimension: {dim_feedforward}")
    print(f"- Dropout: {dropout}")
    print(f"- Learning Rate: {learning_rate}")
    print(f"- Weight Decay: {weight_decay}")
    print(f"- Label Smoothing: {label_smoothing}")
    print(f"- Training Subset Size: {subset_size}")
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Train model
    print("\n=== Training Transformer Model ===")
    model, (train_losses, val_accuracies) = train_model(
        model=model,
        train_batches=train_batches,
        val_batches=val_batches,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        device=device,
        patience=5,
        label_smoothing=label_smoothing
    )
    
    # Print final results
    print(f"\nFinal Results:")
    print(f"- Best Validation Accuracy: {max(val_accuracies):.2f}%")
    print(f"- Final Training Loss: {train_losses[-1]:.4f}")
    
    # Visualize results
    visualize_results(train_losses, val_accuracies)
    print("\nTraining visualization saved to transformer_results.png")

if __name__ == "__main__":
    main()
