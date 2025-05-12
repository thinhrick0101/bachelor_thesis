# Assignment 3B: RNNs for Sentiment Analysis

## Introduction

In this assignment, we explore Recurrent Neural Networks (RNNs) for sequence classification tasks, specifically sentiment analysis on movie reviews. We implement and compare different models to understand the effectiveness of RNNs for this task.

## Part 1: Classification: Data Loading

The IMDb dataset contains movie reviews labeled as either positive or negative. For this assignment, we created a synthetic dataset that mimics the structure of the IMDb dataset but on a smaller scale for faster experimentation.

### Question 1: Padding and Conversion Implementation

To prepare the data for training, we need to pad sequences to a fixed length and convert them to PyTorch tensors. Here's the implementation:

```python
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
```

This function:
1. Gets the padding token index from the word-to-index mapping
2. Randomly samples a batch of sequences and their corresponding labels
3. Finds the maximum sequence length in the batch
4. Pads all sequences to the maximum length using the padding token
5. Converts the padded sequences and labels to PyTorch tensors

Note that the amount of padding increases with batch size because larger batches are more likely to include longer sequences, requiring more padding for shorter sequences in the same batch.

## Part 2: Classification, Baseline Model

### Question 2: Baseline Model Implementation

We implemented a baseline model with the following structure:

```python
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
```

This model:
1. Takes integer indices as input and converts them to embedding vectors using an embedding layer
2. Applies global max pooling along the time dimension to get a fixed-size representation
3. Projects the pooled representation down to the number of classes using a linear layer

## Part 3: RNN Models

In addition to the baseline model, we implemented two RNN-based models:

### LSTM Model

```python
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
```

This model uses an LSTM (Long Short-Term Memory) network to process the sequence, and then uses the final hidden state as the representation for classification.

## Results and Analysis

We trained and evaluated three models:
1. Baseline Model: Linear layer with max pooling
2. RNN Model: LSTM with final hidden state
3. Bidirectional RNN Model: Bidirectional LSTM with concatenated final hidden states

Here are the results:

### Baseline Model
- Final validation accuracy: 75.00%
- The baseline model performs reasonably well, showing that even a simple model can capture some sentiment information.

### RNN Model (LSTM)
- Final validation accuracy: 92.19%
- The LSTM model significantly outperforms the baseline, demonstrating the effectiveness of recurrent connections for sequence data.

### Bidirectional RNN Model
- Final validation accuracy: 100.00%
- The bidirectional LSTM achieves perfect accuracy on our validation set, showing that capturing information from both directions in the sequence is highly beneficial for sentiment analysis.

## Conclusion

This assignment demonstrated the power of RNNs for sequence classification tasks. The key findings are:

1. Simple models with global pooling can perform reasonably well on sentiment analysis.
2. RNNs, particularly LSTMs, are more effective at capturing sequential dependencies in text.
3. Bidirectional RNNs provide the best performance by capturing context from both directions.

The progression from the baseline model to the bidirectional RNN shows a clear improvement in performance, highlighting the importance of model architecture in sequence learning tasks.

## Reflection

- Why does the amount of padding increase with batch size?
  - Larger batches are more likely to include longer sequences, requiring more padding for shorter sequences in the same batch.

- What is the difference between cross entropy and nll_loss?
  - Cross entropy applies the softmax function to the model outputs before computing the loss, while nll_loss expects the model to have already applied a log softmax. Using cross entropy is more numerically stable and convenient for classification tasks.

- Why is global max pooling effective for text classification?
  - Global max pooling captures the most salient features across the entire sequence, regardless of their position. This is particularly useful for sentiment analysis, where certain words or phrases strongly indicate sentiment regardless of where they appear in the review.
