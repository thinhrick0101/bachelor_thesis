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

class MultiHeadSelfAttention(nn.Module):
    """
    Enhanced multi-head self-attention with improved regularization and normalization:
    - Scaling
    - Key/Query/Value projections
    - Multiple attention heads
    - Layer normalization
    - Improved dropout
    """
    def __init__(self, embedding_dim, num_heads=8, dropout=0.1, attention_dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()

        # Ensure embedding_dim is divisible by num_heads
        assert embedding_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        # Store parameters
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # Projections for query, key, and value
        self.query_proj = nn.Linear(embedding_dim, embedding_dim)
        self.key_proj = nn.Linear(embedding_dim, embedding_dim)
        self.value_proj = nn.Linear(embedding_dim, embedding_dim)

        # Output projection
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(embedding_dim)

        # Dropout for attention weights and outputs
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.output_dropout = nn.Dropout(dropout)

        # Scaling factor
        self.scale = math.sqrt(self.head_dim)

        # Initialize weights properly
        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize the projection layers with Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)

        # Initialize biases to zero
        nn.init.constant_(self.query_proj.bias, 0.)
        nn.init.constant_(self.key_proj.bias, 0.)
        nn.init.constant_(self.value_proj.bias, 0.)
        nn.init.constant_(self.output_proj.bias, 0.)

    def forward(self, x, mask=None, return_attention=False):
        """
        Apply multi-head self-attention to the input sequence.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, embedding_dim)
            mask: Optional mask to apply to attention weights
            return_attention: Whether to return attention weights

        Returns:
            Output tensor of shape (batch_size, sequence_length, embedding_dim)
            Attention weights if return_attention=True
        """
        batch_size, seq_length, _ = x.size()

        # Apply layer normalization first (pre-norm approach)
        x_norm = self.layer_norm(x)

        # Project input to query, key, and value
        # (batch_size, seq_length, embedding_dim)
        query = self.query_proj(x_norm)
        key = self.key_proj(x_norm)
        value = self.value_proj(x_norm)

        # Reshape for multi-head attention
        # (batch_size, seq_length, num_heads, head_dim)
        query = query.view(batch_size, seq_length, self.num_heads, self.head_dim)
        key = key.view(batch_size, seq_length, self.num_heads, self.head_dim)
        value = value.view(batch_size, seq_length, self.num_heads, self.head_dim)

        # Transpose to (batch_size, num_heads, seq_length, head_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Compute scaled dot-product attention
        # (batch_size, num_heads, seq_length, seq_length)
        attention_scores = torch.matmul(query, key.transpose(2, 3)) / self.scale

        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply dropout to attention weights
        attention_weights = self.attention_dropout(attention_weights)

        # Compute weighted sum
        # (batch_size, num_heads, seq_length, head_dim)
        context = torch.matmul(attention_weights, value)

        # Transpose back to (batch_size, seq_length, num_heads, head_dim)
        context = context.transpose(1, 2)

        # Reshape to (batch_size, seq_length, embedding_dim)
        context = context.reshape(batch_size, seq_length, self.embedding_dim)

        # Apply output projection
        output = self.output_proj(context)

        # Apply output dropout
        output = self.output_dropout(output)

        # Residual connection
        output = output + x

        if return_attention:
            # Average attention weights across heads for visualization
            avg_attention_weights = attention_weights.mean(dim=1)
            return output, avg_attention_weights
        else:
            return output

class FeedForwardNetwork(nn.Module):
    """
    Feed-forward network used in Transformer models
    """
    def __init__(self, embedding_dim, ffn_dim, dropout=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embedding_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x):
        # Layer normalization
        x_norm = self.layer_norm(x)

        # First linear layer with GELU activation
        x_ffn = self.fc1(x_norm)
        x_ffn = F.gelu(x_ffn)
        x_ffn = self.dropout1(x_ffn)

        # Second linear layer
        x_ffn = self.fc2(x_ffn)
        x_ffn = self.dropout2(x_ffn)

        # Residual connection
        return x + x_ffn

class FullAttentionPoolingModel(nn.Module):
    """
    Enhanced model with embedding, multi-head self-attention, feed-forward network,
    global pooling, and linear projection.
    """
    def __init__(self, vocab_size, embedding_dim, num_classes, num_heads=8,
                 ffn_dim=None, num_layers=2, pooling_type='max', dropout=0.1,
                 attention_dropout=0.1, embedding_dropout=0.1):
        super(FullAttentionPoolingModel, self).__init__()

        # Store embedding dimension
        self.embedding_dim = embedding_dim

        # Set feed-forward dimension if not provided
        if ffn_dim is None:
            ffn_dim = 4 * embedding_dim

        # Embedding layer with dropout
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dropout = nn.Dropout(embedding_dropout)

        # Initialize embedding weights
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)

        # Create a stack of attention and feed-forward layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                MultiHeadSelfAttention(embedding_dim, num_heads=num_heads,
                                      dropout=dropout, attention_dropout=attention_dropout),
                FeedForwardNetwork(embedding_dim, ffn_dim, dropout=dropout)
            ]))

        # Final layer normalization
        self.final_layer_norm = nn.LayerNorm(embedding_dim)

        # Dropout before classification
        self.dropout = nn.Dropout(dropout)

        # Linear projection for classification
        self.linear = nn.Linear(embedding_dim, num_classes)

        # Initialize linear layer
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0.)

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
        embedded = self.embedding_dropout(embedded)

        # Process through attention and feed-forward layers
        attention_weights_list = []
        hidden_state = embedded

        for i, (attention, ffn) in enumerate(self.layers):
            # Apply attention layer
            if return_attention and i == len(self.layers) - 1:  # Only return attention from last layer
                hidden_state, attention_weights = attention(hidden_state, return_attention=True)
                attention_weights_list.append(attention_weights)
            else:
                hidden_state = attention(hidden_state)

            # Apply feed-forward network
            hidden_state = ffn(hidden_state)

        # Apply final layer normalization
        normalized = self.final_layer_norm(hidden_state)

        # Apply pooling
        if self.pooling_type == 'max':
            # Global max pooling
            pooled, _ = torch.max(normalized, dim=1)  # (batch_size, embedding_dim)
        elif self.pooling_type == 'avg':
            # Global average pooling
            pooled = torch.mean(normalized, dim=1)  # (batch_size, embedding_dim)
        elif self.pooling_type == 'sum':
            # Global sum pooling
            pooled = torch.sum(normalized, dim=1)  # (batch_size, embedding_dim)
        elif self.pooling_type == 'cls':
            # Use the first token representation (like BERT's [CLS] token)
            pooled = normalized[:, 0]
        elif self.pooling_type == 'weighted_avg':
            # Weighted average pooling using attention scores
            # Create a learnable weight vector
            if not hasattr(self, 'attention_weights') or self.attention_weights.size(0) != normalized.size(2):
                self.attention_weights = nn.Parameter(torch.ones(normalized.size(2)), requires_grad=True)
                self.attention_weights.data.normal_(mean=0.0, std=0.02)
                # Move to the same device as normalized
                self.attention_weights = self.attention_weights.to(normalized.device)

            # Apply softmax to get attention weights
            attn_weights = F.softmax(self.attention_weights, dim=0)

            # Apply weighted average pooling
            pooled = torch.matmul(normalized, attn_weights)
        elif self.pooling_type == 'hybrid':
            # Hybrid pooling: concatenate max and average pooling
            max_pooled, _ = torch.max(normalized, dim=1)
            avg_pooled = torch.mean(normalized, dim=1)
            pooled = torch.cat([max_pooled, avg_pooled], dim=1)

            # If we're using hybrid pooling, we need to adjust the linear layer
            if not hasattr(self, 'hybrid_projection'):
                self.hybrid_projection = nn.Linear(2 * self.embedding_dim, self.embedding_dim)
                nn.init.xavier_uniform_(self.hybrid_projection.weight)
                nn.init.constant_(self.hybrid_projection.bias, 0.)
                # Move to the same device as the rest of the model
                self.hybrid_projection = self.hybrid_projection.to(normalized.device)

            # Project back to embedding_dim
            pooled = self.hybrid_projection(pooled)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")

        # Apply dropout before classification
        pooled = self.dropout(pooled)

        # Project down to the number of classes
        output = self.linear(pooled)  # (batch_size, num_classes)

        if return_attention and attention_weights_list:
            return output, attention_weights_list[0]  # Return attention from last layer
        else:
            return output

def train_model(model, train_batches, val_batches, num_epochs=5, learning_rate=0.001,
              weight_decay=0.01, warmup_steps=0, device=None, patience=3, label_smoothing=0.0):
    """
    Train the model and evaluate on validation set with improved optimization

    Args:
        model: The model to train
        train_batches: List of training batches
        val_batches: List of validation batches
        num_epochs: Number of training epochs
        learning_rate: Maximum learning rate for the optimizer
        weight_decay: Weight decay for regularization
        warmup_steps: Number of warmup steps for learning rate scheduling
        device: Device to use for training (cuda or cpu)
        patience: Number of epochs to wait for improvement before early stopping
        label_smoothing: Label smoothing factor (0.0 to 1.0)

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

    Args:
        optimizer: The optimizer for which to schedule the learning rate
        num_warmup_steps: The number of steps for the warmup phase
        num_training_steps: The total number of training steps
        last_epoch: The index of the last epoch when resuming training

    Returns:
        Learning rate scheduler
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def find_optimal_learning_rate(vocab_size, embedding_dim, num_classes, num_heads, num_layers,
                              pooling_type, dropout, attention_dropout, embedding_dropout,
                              train_batches, device, start_lr=1e-7, end_lr=1e-2, num_iterations=30):
    """
    Find the optimal learning rate using the learning rate range test.

    Args:
        vocab_size: Size of the vocabulary
        embedding_dim: Dimension of the embedding vectors
        num_classes: Number of output classes
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        pooling_type: Type of pooling
        dropout: Dropout rate
        attention_dropout: Attention dropout rate
        embedding_dropout: Embedding dropout rate
        train_batches: List of training batches
        device: Device to use for training
        start_lr: Starting learning rate
        end_lr: Ending learning rate
        num_iterations: Number of iterations for the test

    Returns:
        Optimal learning rate
    """
    # Create a model for testing
    model = FullAttentionPoolingModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        num_heads=num_heads,
        num_layers=num_layers,
        pooling_type=pooling_type,
        dropout=dropout,
        attention_dropout=attention_dropout,
        embedding_dropout=embedding_dropout
    ).to(device)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Define optimizer with a very low learning rate
    optimizer = optim.AdamW(model.parameters(), lr=start_lr)

    # Learning rate multiplier
    lr_factor = (end_lr / start_lr) ** (1 / num_iterations)

    # Lists to store learning rates and losses
    learning_rates = []
    losses = []

    # Set model to training mode
    model.train()

    # Use a subset of batches for faster testing
    test_batches = train_batches[:min(num_iterations, len(train_batches))]

    # Iterate through batches
    for i, (inputs, targets) in enumerate(test_batches):
        # Move tensors to the configured device
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Store learning rate and loss
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        losses.append(loss.item())

        # Print progress
        if (i + 1) % 10 == 0:
            print(f"Iteration {i+1}/{len(test_batches)}, LR: {current_lr:.2e}, Loss: {loss.item():.4f}")

        # Update parameters
        optimizer.step()

        # Increase learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_factor

    # Find the learning rate with the steepest negative gradient
    min_gradient_idx = 0
    min_gradient = 0

    # Calculate gradients
    for i in range(1, len(learning_rates) - 1):
        gradient = (losses[i+1] - losses[i-1]) / (learning_rates[i+1] - learning_rates[i-1])
        if gradient < min_gradient:
            min_gradient = gradient
            min_gradient_idx = i

    # Get the optimal learning rate (divide by 10 for stability)
    optimal_lr = learning_rates[min_gradient_idx] / 10

    return optimal_lr

def visualize_attention(model, x_val, y_val, i2w, example_idx=0, max_length=50, device=None):
    """
    Visualize attention weights for a specific example

    Args:
        model: Trained model
        x_val: Validation sequences
        y_val: Validation labels
        i2w: Index to word mapping
        example_idx: Index of the example to visualize
        max_length: Maximum sequence length to visualize
        device: Device to use for inference (cuda or cpu)
    """
    # Determine device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set model to evaluation mode
    model.eval()

    # Get example
    example = x_val[example_idx]
    label = y_val[example_idx]

    # Truncate if too long
    if len(example) > max_length:
        example = example[:max_length]

    # Convert to tensor and move to device
    example_tensor = torch.tensor([example], dtype=torch.long).to(device)

    # Get attention weights
    with torch.no_grad():
        _, attention_weights = model(example_tensor, return_attention=True)

    # Move to CPU and convert to numpy
    attention_weights = attention_weights.cpu()[0].numpy()

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
    plt.savefig('full_attention_visualization.png')

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

def apply_data_augmentation(x_train, y_train, i2w, w2i, augmentation_factor=0.3):
    """
    Apply data augmentation techniques to the training data

    Args:
        x_train: Training sequences
        y_train: Training labels
        i2w: Index to word mapping
        w2i: Word to index mapping
        augmentation_factor: Fraction of data to augment

    Returns:
        Augmented training data (x_train_augmented, y_train_augmented)
    """
    import random
    import copy

    # Make a copy of the original data
    x_augmented = copy.deepcopy(x_train)
    y_augmented = copy.deepcopy(y_train)

    # Number of examples to augment
    num_to_augment = int(len(x_train) * augmentation_factor)

    # Randomly select examples to augment
    indices_to_augment = random.sample(range(len(x_train)), num_to_augment)

    # Common words that can be randomly removed without changing meaning much
    stopwords = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'like', 'through', 'over', 'before', 'after']
    stopword_indices = [w2i.get(word, w2i['.unk']) for word in stopwords if word in w2i]

    # Apply different augmentation techniques
    for idx in indices_to_augment:
        sequence = x_train[idx]
        label = y_train[idx]

        # Skip very short sequences
        if len(sequence) < 10:
            continue

        # Choose a random augmentation technique
        technique = random.choice(['random_deletion', 'word_dropout', 'word_swap'])

        if technique == 'random_deletion':
            # Randomly delete some words (especially stopwords)
            new_sequence = []
            for word_idx in sequence:
                # Higher probability of dropping stopwords
                if word_idx in stopword_indices and random.random() < 0.5:
                    continue
                # Lower probability of dropping other words
                elif random.random() < 0.1:
                    continue
                else:
                    new_sequence.append(word_idx)

            # Ensure the sequence is not too short
            if len(new_sequence) >= 5:
                x_augmented.append(new_sequence)
                y_augmented.append(label)

        elif technique == 'word_dropout':
            # Replace some words with [UNK] token
            new_sequence = []
            for word_idx in sequence:
                if random.random() < 0.15:  # 15% chance to replace with UNK
                    new_sequence.append(w2i['.unk'])
                else:
                    new_sequence.append(word_idx)

            x_augmented.append(new_sequence)
            y_augmented.append(label)

        elif technique == 'word_swap':
            # Randomly swap adjacent words
            new_sequence = sequence.copy()
            for i in range(len(new_sequence) - 1):
                if random.random() < 0.15:  # 15% chance to swap
                    new_sequence[i], new_sequence[i+1] = new_sequence[i+1], new_sequence[i]

            x_augmented.append(new_sequence)
            y_augmented.append(label)

    return x_augmented, y_augmented

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

def compare_attention_models(x_train, y_train, x_val, y_val, i2w, w2i, base_batch_size=32, subset_size=10000, use_augmentation=True):
    """
    Compare simple and full-fledged self-attention models

    Args:
        x_train: Training sequences
        y_train: Training labels
        x_val: Validation sequences
        y_val: Validation labels
        i2w: Index to word mapping
        w2i: Word to index mapping
        base_batch_size: Base batch size
        subset_size: Size of the subset to use for training
    """
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Vocabulary size
    vocab_size = len(w2i)
    num_classes = 2

    # Use a subset of the data for faster training
    subset_size = min(subset_size, len(x_train))
    x_train_subset = x_train[:subset_size]
    y_train_subset = y_train[:subset_size]

    # Apply data augmentation if enabled
    if use_augmentation:
        print("\n=== Applying Data Augmentation ===")
        x_train_augmented, y_train_augmented = apply_data_augmentation(x_train_subset, y_train_subset, i2w, w2i)
        print(f"Original training examples: {len(x_train_subset)}")
        print(f"After augmentation: {len(x_train_augmented)}")

        # Use the augmented data for training
        x_train_subset = x_train_augmented
        y_train_subset = y_train_augmented

    # Create batches
    train_batches = imdb_loader.create_batches(x_train_subset, y_train_subset, base_batch_size, w2i)
    val_batches = imdb_loader.create_batches(x_val, y_val, base_batch_size, w2i)

    # Fixed hyperparameters
    num_epochs = 15  # Reasonable number of epochs
    pooling_type = 'avg'  # Simple average pooling works well for sentiment analysis

    # Model hyperparameters - adjusted to prevent overfitting
    embedding_dim = 128  # Reduced embedding dimension to limit capacity
    num_heads = 4  # Fewer attention heads
    num_layers = 1  # Single transformer layer to reduce capacity
    dropout = 0.4  # Increased dropout for stronger regularization
    attention_dropout = 0.3  # Increased attention dropout
    embedding_dropout = 0.2  # Increased embedding dropout

    # Use a fixed learning rate that we know works well
    learning_rate = 1e-4  # Slightly lower learning rate for better generalization
    weight_decay = 0.05  # Increased weight decay for stronger regularization
    warmup_steps = len(train_batches) // 2  # Warmup for half an epoch

    # Add label smoothing
    label_smoothing = 0.1  # Label smoothing factor to prevent overconfidence

    # Create and train the full-fledged self-attention model
    print("\n=== Training Enhanced Self-Attention Model ===")
    full_model = FullAttentionPoolingModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        num_heads=num_heads,
        num_layers=num_layers,
        pooling_type=pooling_type,
        dropout=dropout,
        attention_dropout=attention_dropout,
        embedding_dropout=embedding_dropout
    )

    # Print model architecture and parameter count
    total_params = sum(p.numel() for p in full_model.parameters())
    trainable_params = sum(p.numel() for p in full_model.parameters() if p.requires_grad)
    print(f"Model Parameters: {trainable_params:,} trainable out of {total_params:,} total")

    # Print training configuration
    print(f"\nTraining Configuration:")
    print(f"- Embedding Dimension: {embedding_dim}")
    print(f"- Number of Attention Heads: {num_heads}")
    print(f"- Number of Transformer Layers: {num_layers}")
    print(f"- Pooling Type: {pooling_type}")
    print(f"- Learning Rate: {learning_rate}")
    print(f"- Weight Decay: {weight_decay}")
    print(f"- Dropout: {dropout}")
    print(f"- Attention Dropout: {attention_dropout}")
    print(f"- Embedding Dropout: {embedding_dropout}")
    print(f"- Label Smoothing: {label_smoothing}")
    print(f"- Training Subset Size: {subset_size}")

    # Train the model with improved training procedure
    full_model, (full_losses, full_accuracies) = train_model(
        model=full_model,
        train_batches=train_batches,
        val_batches=val_batches,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        device=device,
        patience=5,  # Early stopping patience
        label_smoothing=label_smoothing  # Add label smoothing
    )

    # Print final results
    print(f"\nFinal Results:")
    print(f"- Best Validation Accuracy: {max(full_accuracies):.2f}%")
    print(f"- Final Training Loss: {full_losses[-1]:.4f}")

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
        print("\n=== Visualizing Full-Fledged Self-Attention ===")
        visualize_attention(full_model, x_val, y_val, i2w, example_idx=example_idx, device=device)
    else:
        # If no examples with negation are found, just use the first example
        visualize_attention(full_model, x_val, y_val, i2w, example_idx=0, device=device)

    print("\nFull-fledged self-attention visualization saved to full_attention_visualization.png")

def main():
    # Load the IMDb dataset
    (x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = imdb_loader.load_imdb_data(final=False)

    # Print dataset information
    print(f'Training examples: {len(x_train)}, Validation examples: {len(x_val)}')
    print(f'Vocabulary size: {len(i2w)}, Number of classes: {numcls}')

    # Compare attention models
    compare_attention_models(x_train, y_train, x_val, y_val, i2w, w2i)

if __name__ == "__main__":
    main()

