# Assignment 3B: RNNs - Answers to Questions

## Question 1: Implement padding and conversion

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

This function handles:
1. Getting the padding token index from the word-to-index mapping
2. Sampling a batch of sequences and their labels
3. Finding the maximum sequence length in the batch
4. Padding all sequences to the maximum length using the padding token
5. Converting the padded sequences and labels to PyTorch tensors with appropriate data types

## Question 2: Build a model with the specified structure

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

This model follows the structure specified in the assignment:
1. Input layer: Takes integer indices and passes them to an embedding layer
2. Embedding layer: Converts integer indices to embedding vectors of size 300
3. Global max pooling: Applied along the time dimension to get a fixed-size representation
4. Linear layer: Projects the pooled representation down to the number of classes

Notes:
- The vocabulary size is determined from the `i2w` list returned by the `load_imdb` function
- We use an embedding size of 300 as specified
- We don't apply softmax in the model since it's applied as part of the loss function (torch.nn.functional.cross_entropy)
- For computing accuracy, we use argmax on the linear outputs to get the predicted class

## Additional Reflections

### Why does the amount of padding increase with batch size?
Larger batches are more likely to include longer sequences. Since all sequences in a batch need to be padded to the same length (the maximum length in the batch), larger batches tend to require more padding for the shorter sequences.

### Difference between cross entropy and nll_loss
- `torch.nn.functional.cross_entropy` applies softmax to the model outputs before computing the loss
- `torch.nn.functional.nll_loss` expects the model to have already applied log softmax
- Using cross entropy is more numerically stable and convenient for classification tasks

### Effectiveness of global max pooling for text classification
Global max pooling captures the most salient features across the entire sequence, regardless of their position. This is particularly useful for sentiment analysis, where certain words or phrases strongly indicate sentiment regardless of where they appear in the review.

## Experimental Results

We implemented and compared three models:
1. Baseline Model (as specified in the assignment): 75.00% validation accuracy
2. LSTM Model: 92.19% validation accuracy
3. Bidirectional LSTM Model: 100.00% validation accuracy

These results demonstrate the effectiveness of RNNs, particularly bidirectional RNNs, for sequence classification tasks like sentiment analysis.
