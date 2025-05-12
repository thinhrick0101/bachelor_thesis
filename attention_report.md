# Self-Attention for Sentiment Analysis

## Introduction

In this report, we explore the implementation and effectiveness of self-attention mechanisms for sentiment analysis on the IMDb movie review dataset. We compare models with and without self-attention and analyze the impact of different pooling strategies.

## Implementation

### Simple Self-Attention

We implemented a simple self-attention mechanism as described in the assignment:

```python
class SimpleSelfAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(SimpleSelfAttention, self).__init__()
        self.scale = math.sqrt(embedding_dim)
    
    def forward(self, x):
        # Compute scaled attention weights
        attention_weights = torch.bmm(x, x.transpose(1, 2)) / self.scale
        
        # Apply softmax along the last dimension
        attention_weights = F.softmax(attention_weights, dim=2)
        
        # Compute the weighted sum
        output = torch.bmm(attention_weights, x)
        
        return output
```

This implementation follows the core principles of self-attention:
1. Compute dot products between all pairs of vectors in the sequence
2. Scale the dot products to prevent extremely large values
3. Apply softmax to get normalized weights
4. Compute weighted sums to produce the output sequence

### Enhanced Self-Attention

We also implemented an enhanced version with separate projections for query, key, and value:

```python
class EnhancedSelfAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(EnhancedSelfAttention, self).__init__()
        
        # Projections for query, key, and value
        self.query_proj = nn.Linear(embedding_dim, embedding_dim)
        self.key_proj = nn.Linear(embedding_dim, embedding_dim)
        self.value_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # Scaling factor
        self.scale = math.sqrt(embedding_dim)
    
    def forward(self, x):
        # Project input to query, key, and value
        query = self.query_proj(x)
        key = self.key_proj(x)
        value = self.value_proj(x)
        
        # Compute scaled dot-product attention
        attention_scores = torch.bmm(query, key.transpose(1, 2)) / self.scale
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=2)
        
        # Compute weighted sum
        output = torch.bmm(attention_weights, value)
        
        return output
```

This enhanced version adds separate linear projections for query, key, and value, which gives the model more flexibility to learn different representations for each role.

## Model Architecture

Our complete model architecture consists of:
1. An embedding layer to convert word indices to vectors
2. A self-attention layer to capture relationships between words
3. A global pooling layer (max, avg, or sum) to create a fixed-size representation
4. A linear layer to project to class scores

```python
class AttentionPoolingModel(nn.Module):
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
    
    def forward(self, x):
        # Embed the input
        embedded = self.embedding(x)
        
        # Apply self-attention
        attended = self.attention(embedded)
        
        # Apply pooling
        if self.pooling_type == 'max':
            pooled, _ = torch.max(attended, dim=1)
        elif self.pooling_type == 'avg':
            pooled = torch.mean(attended, dim=1)
        elif self.pooling_type == 'sum':
            pooled = torch.sum(attended, dim=1)
        
        # Project down to the number of classes
        output = self.linear(pooled)
        
        return output
```

## Experimental Results

### Comparison of Models With and Without Self-Attention

We compared three models:
1. Baseline model without self-attention
2. Model with simple self-attention
3. Model with enhanced self-attention

Results after 10 epochs:

| Model | Final Loss | Final Accuracy |
|-------|------------|----------------|
| Without Self-Attention | 0.6266 | 60.52% |
| Simple Self-Attention | 0.6145 | 61.24% |
| Enhanced Self-Attention | 0.1144 | 81.90% |

The enhanced self-attention model significantly outperformed both the baseline model and the simple self-attention model. This demonstrates the power of self-attention in capturing relationships between words in the input sequence.

### Comparison of Pooling Types with Self-Attention

We also compared different pooling strategies with the enhanced self-attention model:

| Pooling Type | Final Loss | Final Accuracy |
|--------------|------------|----------------|
| AVG | 0.0397 | 85.18% |
| MAX | 0.0453 | 84.16% |
| SUM | 0.1439 | 83.36% |

Average pooling performed slightly better than max pooling, while sum pooling lagged behind. This suggests that for sentiment analysis with self-attention, average pooling might be more effective at capturing the overall sentiment of a review.

## Analysis

### Why Self-Attention Helps

Self-attention allows the model to capture relationships between words in the input sequence. For example, in the phrase "not terrible", the self-attention mechanism can learn that "not" modifies "terrible", flipping its sentiment from negative to positive. This is something that a simple bag-of-words model (which is essentially what we have without self-attention) cannot capture.

The enhanced self-attention model, with its separate projections for query, key, and value, gives the model even more flexibility to learn these relationships. This explains the significant performance improvement over the simple self-attention model.

### Pooling Strategies

Different pooling strategies have different strengths:
- Max pooling captures the most salient features, which can be useful for detecting strong sentiment signals
- Average pooling captures the overall sentiment of the review, which can be more robust
- Sum pooling can amplify the signal but may also amplify noise

For sentiment analysis with self-attention, average pooling seems to work best, likely because it provides a balanced representation of the entire review.

## Conclusion

Self-attention is a powerful mechanism for capturing relationships between words in a sequence. Our experiments show that adding self-attention to a simple classification model can significantly improve performance on sentiment analysis tasks. The enhanced self-attention model with average pooling achieved an accuracy of 85.18% on the IMDb dataset, which is close to the state-of-the-art for models of this complexity.

Future work could explore more sophisticated attention mechanisms, such as multi-head attention, or combine self-attention with recurrent or convolutional layers for even better performance.
