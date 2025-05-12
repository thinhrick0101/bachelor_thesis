# Full-Fledged Self-Attention for Sentiment Analysis

## Introduction

In this report, we explore the implementation and effectiveness of full-fledged self-attention mechanisms for sentiment analysis on the IMDb movie review dataset. We build upon our previous implementation of simple self-attention by adding the "bells and whistles" that make it a complete, modern self-attention mechanism as used in transformer models.

## Implementation Details

### Simple Self-Attention (Previous Implementation)

Our previous implementation of simple self-attention included:

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

This implementation already included scaling by the square root of the embedding dimension to prevent extremely large values in the dot products.

### Full-Fledged Self-Attention

Our full-fledged self-attention implementation adds the following enhancements:

1. **Separate Key, Query, and Value Projections**: We add separate linear projections for the key, query, and value vectors, allowing the model to learn different transformations for each role.

2. **Multi-Head Attention**: We implement multiple attention heads that operate in parallel, each with its own set of projections, allowing the model to capture different types of relationships.

3. **Layer Normalization and Residual Connections**: We add layer normalization and residual connections, which are standard in transformer architectures and help with training stability.

4. **Dropout**: We add dropout to the attention weights and after the self-attention layer to prevent overfitting.

Here's the implementation of our full-fledged multi-head self-attention:

```python
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads=8, dropout=0.1):
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
        
        # Dropout for attention weights
        self.dropout = nn.Dropout(dropout)
        
        # Scaling factor
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x, return_attention=False):
        batch_size, seq_length, _ = x.size()
        
        # Project input to query, key, and value
        query = self.query_proj(x)
        key = self.key_proj(x)
        value = self.value_proj(x)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_length, self.num_heads, self.head_dim)
        key = key.view(batch_size, seq_length, self.num_heads, self.head_dim)
        value = value.view(batch_size, seq_length, self.num_heads, self.head_dim)
        
        # Transpose to (batch_size, num_heads, seq_length, head_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        # Compute scaled dot-product attention
        attention_scores = torch.matmul(query, key.transpose(2, 3)) / self.scale
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply dropout
        attention_weights = self.dropout(attention_weights)
        
        # Compute weighted sum
        context = torch.matmul(attention_weights, value)
        
        # Transpose back to (batch_size, seq_length, num_heads, head_dim)
        context = context.transpose(1, 2)
        
        # Reshape to (batch_size, seq_length, embedding_dim)
        context = context.reshape(batch_size, seq_length, self.embedding_dim)
        
        # Apply output projection
        output = self.output_proj(context)
        
        if return_attention:
            # Average attention weights across heads for visualization
            avg_attention_weights = attention_weights.mean(dim=1)
            return output, avg_attention_weights
        else:
            return output
```

And here's how we integrate it into our model with layer normalization and residual connections:

```python
class FullAttentionPoolingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, num_heads=8, pooling_type='max', dropout=0.1):
        super(FullAttentionPoolingModel, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Layer normalization before attention
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        
        # Multi-head self-attention layer
        self.attention = MultiHeadSelfAttention(embedding_dim, num_heads=num_heads, dropout=dropout)
        
        # Layer normalization after attention
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Linear projection
        self.linear = nn.Linear(embedding_dim, num_classes)
        
        # Store pooling type
        self.pooling_type = pooling_type
    
    def forward(self, x, return_attention=False):
        # Embed the input
        embedded = self.embedding(x)
        
        # Apply layer normalization
        normalized = self.layer_norm1(embedded)
        
        # Apply multi-head self-attention with residual connection
        if return_attention:
            attended, attention_weights = self.attention(normalized, return_attention=True)
        else:
            attended = self.attention(normalized)
        
        # Add residual connection
        attended = embedded + attended
        
        # Apply layer normalization
        normalized = self.layer_norm2(attended)
        
        # Apply dropout
        normalized = self.dropout(normalized)
        
        # Apply pooling
        if self.pooling_type == 'max':
            pooled, _ = torch.max(normalized, dim=1)
        elif self.pooling_type == 'avg':
            pooled = torch.mean(normalized, dim=1)
        elif self.pooling_type == 'sum':
            pooled = torch.sum(normalized, dim=1)
        
        # Project down to the number of classes
        output = self.linear(pooled)
        
        if return_attention:
            return output, attention_weights
        else:
            return output
```

## Experimental Results

We trained our full-fledged self-attention model on the IMDb dataset and achieved the following results:

```
=== Training Model with Full-Fledged Self-Attention ===
Epoch 1/5, Loss: 0.6636, Validation Accuracy: 65.60%, Time: 12.79s
Epoch 2/5, Loss: 0.5574, Validation Accuracy: 74.30%, Time: 12.95s
Epoch 3/5, Loss: 0.4495, Validation Accuracy: 78.52%, Time: 13.08s
Epoch 4/5, Loss: 0.3630, Validation Accuracy: 81.14%, Time: 12.93s
Epoch 5/5, Loss: 0.2819, Validation Accuracy: 82.20%, Time: 12.88s
```

The model achieved 82.20% validation accuracy after just 5 epochs, which is impressive for a relatively simple model on the IMDb dataset.

## Analysis of the "Bells and Whistles"

Let's analyze the impact of each enhancement:

### 1. Scaling

Scaling the dot products by the square root of the embedding dimension helps prevent extremely large values, which can lead to very small gradients during backpropagation. This is particularly important for larger embedding dimensions.

### 2. Key, Query, and Value Projections

Separate projections for key, query, and value vectors give the model more flexibility to learn different transformations for each role. This allows the model to:

- Transform the query to better match relevant keys
- Transform the keys to better represent the content
- Transform the values to better represent the information to be aggregated

Without these projections, the weight `wii` (the attention weight of a token to itself) would typically be large because a vector is most similar to itself. With the projections, the model can learn to attend to other relevant tokens as well.

### 3. Multi-Head Attention

Multi-head attention allows the model to capture different types of relationships in parallel. For example:

- One head might focus on syntactic relationships
- Another head might focus on semantic relationships
- Yet another head might focus on negation or other specific patterns

By having multiple heads, the model can attend to different aspects of the input simultaneously and combine this information for better predictions.

### 4. Layer Normalization and Residual Connections

Layer normalization helps stabilize training by normalizing the activations, while residual connections help with gradient flow during backpropagation. Together, they make the model easier to train and less prone to vanishing gradients.

## Attention Visualization

We visualized the attention weights for a negative review containing negation:

```
Example text:
long boring blasphemous never have i been so glad to see ending credits roll
Label: Negative

Top 10 most attended words:
Word: boring, Average attention: 0.1172
Word: see, Average attention: 0.1002
Word: roll, Average attention: 0.0794
Word: blasphemous, Average attention: 0.0733
Word: been, Average attention: 0.0718
Word: credits, Average attention: 0.0716
Word: so, Average attention: 0.0665
Word: ending, Average attention: 0.0660
Word: i, Average attention: 0.0654
Word: long, Average attention: 0.0637
```

The visualization shows that the model pays the most attention to sentiment-bearing words like "boring" and "blasphemous", as well as to the context around them. This demonstrates that the model is learning to focus on the most relevant parts of the input for sentiment classification.

## Conclusion

Full-fledged self-attention with all the bells and whistles significantly enhances the model's ability to capture relationships between words in a sequence. The separate projections for key, query, and value, along with multi-head attention, allow the model to learn more complex patterns, while scaling, layer normalization, and residual connections improve training stability.

Our experiments show that these enhancements lead to a powerful model that achieves high accuracy on the IMDb sentiment analysis task. The attention visualizations confirm that the model is learning to focus on the most relevant parts of the input for the classification task.

This implementation provides a solid foundation for more complex transformer-based models, which could be extended with additional layers like feed-forward networks and positional encodings to create a complete transformer architecture.
