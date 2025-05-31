"""
Sparse attention implementation with efficient masking and caching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SparseMultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embedding_dim // num_heads
        assert self.head_dim * num_heads == embedding_dim, "embedding_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        
        self.cache = {}
        self.scale = math.sqrt(self.head_dim)
    
    def _shape(self, tensor, seq_len, bsz):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    
    def forward(self, x, mask=None, return_attention=False):
        """
        Args:
            x: Input of shape [batch_size, seq_len, embedding_dim]
            mask: Boolean mask of shape [batch_size, seq_len, seq_len]
            return_attention: Whether to return attention weights
        """
        bsz, seq_len, _ = x.shape
        
        # Project and reshape
        q = self._shape(self.q_proj(x), seq_len, bsz)  # [B, H, L, D]
        k = self._shape(self.k_proj(x), seq_len, bsz)  # [B, H, L, D]
        v = self._shape(self.v_proj(x), seq_len, bsz)  # [B, H, L, D]
        
        # Compute attention scores
        attention_scores = torch.matmul(q, k.transpose(2, 3)) / self.scale  # [B, H, L, L]
        
        # Apply mask if provided
        if mask is not None:
            # Use a smaller negative value that works with float16
            mask_value = -65504.0 if attention_scores.dtype == torch.float16 else -1e9
            attention_scores = attention_scores.masked_fill(~mask.unsqueeze(1), mask_value)
        
        # Apply softmax and dropout
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = F.dropout(attention_probs, p=self.dropout, training=self.training)
        
        # Get output
        output = torch.matmul(attention_probs, v)  # [B, H, L, D]
        output = output.transpose(1, 2).reshape(bsz, seq_len, self.embedding_dim)
        output = self.out_proj(output)
        
        if return_attention:
            return output, attention_probs
        return output


class SparseTransformerLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_dim, dropout=0.0, attention_dropout=0.0):
        super().__init__()
        
        # First normalization and attention
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attention = SparseMultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=attention_dropout
        )
        self.dropout1 = nn.Dropout(dropout)
        
        # Second normalization and FFN
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embedding_dim)
        )
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None, return_attention=False):
        # First sub-layer: Multi-head attention
        x2 = self.norm1(x)
        if return_attention:
            attn_out, attn_weights = self.attention(x2, mask, return_attention=True)
            x = x + self.dropout1(attn_out)
            return x, attn_weights
        else:
            x = x + self.dropout1(self.attention(x2, mask))
        
        # Second sub-layer: FFN
        x = x + self.dropout2(self.ffn(self.norm2(x)))
        return x

class SparseTransformer(nn.Module):
    """
    Full transformer model with sparse attention
    """
    def __init__(self, vocab_size, embedding_dim, num_classes, num_heads=8,
                 num_layers=6, ffn_dim=None, dropout=0.1, attention_dropout=0.1):
        super(SparseTransformer, self).__init__()
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            SparseTransformerLayer(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                attention_dropout=attention_dropout
            )
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        # Initialize embedding
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        
        # Initialize classifier
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x, mask=None, return_attention=False):
        # Get embeddings
        x = self.embedding(x)
        x = self.embedding_dropout(x)
        
        attention_weights = []
        
        # Pass through transformer layers
        for layer in self.layers:
            if return_attention:
                x, attn = layer(x, mask, return_attention=True)
                attention_weights.append(attn)
            else:
                x = layer(x, mask)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Classification
        logits = self.classifier(x)
        
        if return_attention:
            return logits, attention_weights
        return logits 