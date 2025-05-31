"""
Byte-level transformer model with sparse attention patterns for Enwik8 dataset.
This is a direct replacement for EnhancedCharTransformer, implementing static sparse attention
patterns derived from cluster analysis of dense attention heads.
"""

import torch
import torch.nn as nn
import math
from sparse_attention import SparseMultiHeadAttention, SparseTransformerLayer
from stable_char_transformer import ImprovedPositionalEncoding


class SparseByteTransformer(nn.Module):
    """
    Byte-level language model using sparse transformer for Enwik8 dataset.
    This is a direct replacement for EnhancedCharTransformer, using sparse attention.
    
    The model uses:
    - Fixed vocabulary size of 256 (byte-level)
    - Scaled token embeddings
    - Improved positional encoding
    - Token-level dropout
    - Stack of sparse transformer layers
    - Weight-tied output projection
    """
    
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_layers=12,
        dim_feedforward=2048,
        dropout=0.1,
        attention_dropout=0.1,
        token_dropout=0.0,
        max_len=5000
    ):
        """
        Initialize the transformer model.
        
        Args:
            d_model: Dimension of the model
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of the feedforward network
            dropout: Dropout rate
            attention_dropout: Dropout rate for attention weights
            token_dropout: Probability of dropping entire token embeddings
            max_len: Maximum sequence length for positional encoding
        """
        super(SparseByteTransformer, self).__init__()
        
        # Model dimensions
        self.vocab_size = 256  # Fixed for byte-level modeling
        self.d_model = d_model
        self.token_dropout = token_dropout
        
        # Token embedding with scaling
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.embed_scale = math.sqrt(d_model)
        
        # Positional encoding
        self.pos_encoder = ImprovedPositionalEncoding(
            d_model=d_model,
            max_len=max_len,
            dropout=dropout
        )
        
        # Dropout layers
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Stack of transformer layers
        self.layers = nn.ModuleList([
            SparseTransformerLayer(
                embedding_dim=d_model,
                num_heads=nhead,
                ffn_dim=dim_feedforward,
                dropout=dropout,
                attention_dropout=attention_dropout
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(d_model)
        
        # Output projection to vocabulary (tied with embedding)
        self.output_projection = nn.Linear(self.vocab_size, d_model, bias=False)
        self.output_projection.weight = self.embedding.weight
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the embedding weights with normal distribution."""
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        # Output projection weights are tied to embedding
    
    def _apply_token_dropout(self, x, training=True):
        """
        Randomly mask out entire token embeddings with probability token_dropout.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, d_model]
            training: Whether the model is in training mode
        
        Returns:
            x: Tensor with randomly masked token embeddings
        """
        if training and self.token_dropout > 0:
            mask = torch.bernoulli(
                torch.full(x.shape[:2], 1 - self.token_dropout, device=x.device)
            ).unsqueeze(-1)
            x = x * mask
        return x
    
    def forward(self, x, return_attention=False):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length]
            return_attention: Whether to return attention maps for visualization
        
        Returns:
            logits: Output logits of shape [batch_size, seq_length, vocab_size]
            attention_maps: Optional list of attention maps from each layer
        """
        batch_size, seq_length = x.size()
        
        # Create causal mask for autoregressive modeling
        causal_mask = torch.triu(
            torch.ones(seq_length, seq_length, device=x.device),
            diagonal=1
        ).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]
        
        # 1. Token embedding with scaling
        x = self.embedding(x) * self.embed_scale  # [B, L, d_model]
        
        # 2. Add positional encoding
        x = self.pos_encoder(x)
        
        # 3. Apply embedding dropout
        x = self.embedding_dropout(x)
        
        # 4. Apply token dropout if enabled
        x = self._apply_token_dropout(x, self.training)
        
        attention_maps = []
        
        # 5. Pass through transformer layers
        for layer in self.layers:
            if return_attention:
                x, attn = layer(x, mask=~causal_mask, return_attention=True)
                attention_maps.append(attn)
            else:
                x = layer(x, mask=~causal_mask)
        
        # 6. Final layer norm
        x = self.norm(x)
        
        # 7. Project to vocabulary size
        logits = self.output_projection(x)
        
        if return_attention:
            return logits, attention_maps
        return logits
    
    def generate(self, prefix_tokens, max_new_tokens, temperature=1.0):
        """
        Generate new tokens autoregressively given a prefix.
        
        Args:
            prefix_tokens: Starting sequence [batch_size, prefix_length]
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature (1.0 = standard, <1.0 = more focused)
        
        Returns:
            tokens: Generated sequence [batch_size, prefix_length + max_new_tokens]
        """
        self.eval()
        with torch.no_grad():
            # Start with the prefix
            tokens = prefix_tokens.clone()
            
            # Generate new tokens one at a time
            for _ in range(max_new_tokens):
                # Get predictions
                logits = self(tokens)  # [B, L, 256]
                
                # Only take the last token's predictions
                next_token_logits = logits[:, -1, :] / temperature  # [B, 256]
                
                # Sample from the distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
                
                # Append to the sequence
                tokens = torch.cat([tokens, next_token], dim=1)
            
            return tokens