import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import imdb_data
import random
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

def create_batch(sequences, labels, batch_size, w2i):
    """
    Create a batch of padded sequences and corresponding labels.

    Args:
        sequences: List of sequences (each sequence is a list of word indices)
        labels: List of labels for each sequence
        batch_size: Size of the batch
        w2i: Word to index mapping dictionary

    Returns:
        Tuple of (padded_batch, labels_batch) as PyTorch tensors
    """
    # Get the padding token index
    pad_idx = w2i[".pad"]

    # Randomly sample batch_size sequences
    indices = random.sample(range(len(sequences)), min(batch_size, len(sequences)))
    batch_sequences = [sequences[i] for i in indices]
    batch_labels = [labels[i] for i in indices]

    # Find the maximum length in this batch
    max_length = max(len(seq) for seq in batch_sequences)

    # Pad all sequences to the maximum length
    padded_batch = []
    for seq in batch_sequences:
        # Pad the sequence with the padding token
        padded_seq = seq + [pad_idx] * (max_length - len(seq))
        padded_batch.append(padded_seq)

    # Convert to PyTorch tensors
    padded_batch = torch.tensor(padded_batch, dtype=torch.long)
    labels_batch = torch.tensor(batch_labels, dtype=torch.long)

    return padded_batch, labels_batch

# Define the baseline model (Linear layer with max pooling)
class BaselineModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes):
        super(BaselineModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        embedded = self.embedding(x)
        # embedded shape: (batch_size, sequence_length, embedding_dim)

        # Global max pooling along the time dimension
        pooled, _ = torch.max(embedded, dim=1)
        # pooled shape: (batch_size, embedding_dim)

        # Project down to the number of classes
        output = self.linear(pooled)
        # output shape: (batch_size, num_classes)

        return output

# Define an RNN model
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers=1, bidirectional=False):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )
        # If bidirectional, we need to account for the doubled hidden dimension
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        batch_size = x.size(0)

        # Embed the input
        embedded = self.embedding(x)
        # embedded shape: (batch_size, sequence_length, embedding_dim)

        # Pass through RNN
        rnn_out, (hidden, _) = self.rnn(embedded)
        # rnn_out shape: (batch_size, sequence_length, hidden_dim * num_directions)
        # hidden shape: (num_layers * num_directions, batch_size, hidden_dim)

        # Get the final hidden state
        if self.bidirectional:
            # Concatenate the last hidden state from both directions
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1]

        # Pass through the fully connected layer
        output = self.fc(hidden)
        # output shape: (batch_size, num_classes)

        return output

def train_model(model, x_train, y_train, x_val, y_val, w2i, batch_size=32, num_epochs=5, learning_rate=0.001):
    """
    Train the model and evaluate on validation set
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

        # Create random batches for training
        for _ in range(0, len(x_train), batch_size):
            # Get a batch
            inputs, targets = create_batch(x_train, y_train, batch_size, w2i)

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
            for i in range(0, len(x_val), batch_size):
                # Get a batch
                inputs, targets = create_batch(x_val, y_val, batch_size, w2i)

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

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.2f}%')

    # Plot training metrics
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')

    plt.tight_layout()
    plt.savefig('training_metrics.png')

    return model

def main():
    # Load the IMDB dataset
    (x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = imdb_data.load_imdb(final=False)

    # Print dataset information
    print(f'Training examples: {len(x_train)}, Validation examples: {len(x_val)}')
    print(f'Vocabulary size: {len(i2w)}, Number of classes: {numcls}')
    print(f'Example review: {[i2w[w] for w in x_train[0][:20]]}...')

    # Model parameters
    vocab_size = len(i2w)
    embedding_dim = 300
    hidden_dim = 128

    # Create models
    baseline_model = BaselineModel(vocab_size, embedding_dim, numcls)
    rnn_model = RNNModel(vocab_size, embedding_dim, hidden_dim, numcls)
    bidirectional_rnn_model = RNNModel(vocab_size, embedding_dim, hidden_dim, numcls, bidirectional=True)

    # Train and evaluate the baseline model
    print("\n=== Training Baseline Model ===")
    baseline_trained = train_model(baseline_model, x_train, y_train, x_val, y_val, w2i, num_epochs=10)

    # Train and evaluate the RNN model
    print("\n=== Training RNN Model ===")
    rnn_trained = train_model(rnn_model, x_train, y_train, x_val, y_val, w2i, num_epochs=10)

    # Train and evaluate the bidirectional RNN model
    print("\n=== Training Bidirectional RNN Model ===")
    birnn_trained = train_model(bidirectional_rnn_model, x_train, y_train, x_val, y_val, w2i, num_epochs=10)

    print("\nTraining completed!")
    print("Models comparison:")
    print("1. Baseline Model: Linear layer with max pooling")
    print("2. RNN Model: LSTM with final hidden state")
    print("3. Bidirectional RNN Model: Bidirectional LSTM with concatenated final hidden states")
    print("\nCheck training_metrics.png for visualization of training progress.")

if __name__ == "__main__":
    main()
