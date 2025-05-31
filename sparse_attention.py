import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SparseMultiHeadAttention(nn.Module):
    """
    Multi-head attention with sparse attention patterns based on cluster analysis.
    Each head uses a different sparse pattern based on its cluster assignment.
    """
    def __init__(self, embedding_dim, num_heads=8, dropout=0.1, attention_dropout=0.1):
        super(SparseMultiHeadAttention, self).__init__()
        
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
        
        # Dropout
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.output_dropout = nn.Dropout(dropout)
        
        # Scaling factor
        self.scale = math.sqrt(self.head_dim)
        
        # Initialize weights
        self._reset_parameters()
        
        # Cluster assignments for each head
        self.cluster_assignments = [0, 1, 2, 3, 0, 1, 2, 3]  # For 8 heads
        
        # Cache for sparse attention masks
        self._mask_cache = {}

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)
        
        nn.init.constant_(self.query_proj.bias, 0.)
        nn.init.constant_(self.key_proj.bias, 0.)
        nn.init.constant_(self.value_proj.bias, 0.)
        nn.init.constant_(self.output_proj.bias, 0.)
    
    def _create_sparse_mask(self, seq_length):
        """
        Internal method to create sparse attention masks.
        Creates masks on CPU for caching efficiency.
        """
        # Initialize mask: [num_heads, seq_length, seq_length]
        mask = torch.zeros(self.num_heads, seq_length, seq_length)
        
        for h in range(self.num_heads):
            cluster = self.cluster_assignments[h]
            
            for i in range(seq_length):
                if cluster == 0:  # Narrow local window (±8)
                    start_idx = max(0, i - 8)
                    end_idx = min(seq_length, i + 9)
                    mask[h, i, start_idx:end_idx] = 1
                
                elif cluster == 3:  # Wide local window (±16)
                    start_idx = max(0, i - 16)
                    end_idx = min(seq_length, i + 17)
                    mask[h, i, start_idx:end_idx] = 1
                
                elif cluster == 1:  # Local + Strided
                    # Local window (±16)
                    start_idx = max(0, i - 16)
                    end_idx = min(seq_length, i + 17)
                    mask[h, i, start_idx:end_idx] = 1
                    # Strided attention (every 8th token)
                    mask[h, i, ::8] = 1
                
                elif cluster == 2:  # Global anchors
                    # Only apply strided attention if sequence is long enough
                    if seq_length >= 32:
                        mask[h, i, ::32] = 1
                    
                    # Global anchor positions
                    global_pos = [
                        0,  # Start
                        seq_length // 2,  # Middle
                        seq_length - 1,  # End
                    ]
                    mask[h, i, global_pos] = 1
        
        # Convert to boolean tensor
        return mask > 0

    def get_sparse_mask(self, seq_length, device):
        """
        Get cached sparse attention mask or create a new one.
        Masks are stored on CPU and moved to the correct device when needed.
        """
        if seq_length not in self._mask_cache:
            mask = self._create_sparse_mask(seq_length)
            self._mask_cache[seq_length] = mask
        
        # Move mask to the correct device
        return self._mask_cache[seq_length].to(device)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, _ = x.size()
        
        # Project input to query, key, and value
        query = self.query_proj(x)
        key = self.key_proj(x)
        value = self.value_proj(x)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_length, self.num_heads, self.head_dim)
        key = key.view(batch_size, seq_length, self.num_heads, self.head_dim)
        value = value.view(batch_size, seq_length, self.num_heads, self.head_dim)
        
        # Transpose to [batch_size, num_heads, seq_length, head_dim]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        # Compute scaled dot-product attention
        attention_scores = torch.matmul(query, key.transpose(2, 3)) / self.scale
        
        # Get cached sparse attention mask
        sparse_mask = self.get_sparse_mask(seq_length, x.device)
        attention_scores = attention_scores.masked_fill(~sparse_mask.unsqueeze(0), -1e9)
        
        # Apply additional mask if provided (e.g., padding mask)
        if mask is not None:
            # Ensure mask has correct shape for broadcasting
            if mask.dim() == 2:  # [B, L]
                mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
            elif mask.dim() == 3:  # [B, 1, L]
                mask = mask.unsqueeze(1)  # [B, 1, 1, L]
            attention_scores = attention_scores.masked_fill(~mask, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply dropout to attention weights
        attention_weights = self.attention_dropout(attention_weights)
        
        # Compute weighted sum
        context = torch.matmul(attention_weights, value)
        
        # Transpose back to [batch_size, seq_length, num_heads, head_dim]
        context = context.transpose(1, 2)
        
        # Reshape to [batch_size, seq_length, embedding_dim]
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

class SparseTransformerLayer(nn.Module):
    """
    Transformer layer with sparse multi-head attention using pre-norm architecture
    """
    def __init__(self, embedding_dim, num_heads=8, ffn_dim=None, dropout=0.1, attention_dropout=0.1):
        super(SparseTransformerLayer, self).__init__()
        
        if ffn_dim is None:
            ffn_dim = 4 * embedding_dim
        
        # Sparse multi-head attention
        self.attention = SparseMultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            attention_dropout=attention_dropout
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embedding_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        
    def forward(self, x, mask=None, return_attention=False):
        # Pre-norm for attention
        x2 = self.norm1(x)
        
        # Multi-head attention block
        if return_attention:
            attn_out, attention_weights = self.attention(x2, mask, return_attention=True)
        else:
            attn_out = self.attention(x2, mask)
        
        # First residual connection
        x = x + attn_out
        
        # Pre-norm for FFN
        x2 = self.norm2(x)
        
        # Feed-forward block with residual
        x = x + self.ffn(x2)
        
        if return_attention:
            return x, attention_weights
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