"""
Optimized sparse attention implementation with block-sparse patterns and efficient memory usage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import warnings

try:
    import xformers.ops as xops
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xformers not available. Falling back to standard implementation.")

class SparseMultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embedding_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        assert self.head_dim * num_heads == embedding_dim, "embedding_dim must be divisible by num_heads"
        assert num_heads % 4 == 0, "num_heads must be divisible by 4 (we divide heads into 4 clusters)"

        self.heads_per_cluster = num_heads // 4

        # Q/K/V projections for all heads at once
        self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)

        # Initialize sparse attention blocks configuration
        self._init_sparse_config()
        
        # Cache for block-sparse masks
        self._mask_cache = {}

    def _init_sparse_config(self):
        """Initialize block-sparse attention configuration"""
        self.block_size = 32  # Optimize for GPU memory access
        self.sparsity_config = {
            0: {'window': 8, 'stride': 0},    # Local narrow
            1: {'window': 32, 'stride': 8},   # Local wide + strided
            2: {'window': 0, 'stride': 32},   # Global + strided
            3: {'window': 32, 'stride': 0}    # Local wide
        }

    def _create_block_sparse_layout(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create block-sparse attention layout optimized for GPU computation
        Returns a boolean tensor of shape [num_heads, num_blocks, num_blocks]
        """
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        layout = torch.zeros(self.num_heads, num_blocks, num_blocks, dtype=torch.bool, device=device)
        
        for cluster_idx in range(4):
            start_h = cluster_idx * self.heads_per_cluster
            end_h = (cluster_idx + 1) * self.heads_per_cluster
            config = self.sparsity_config[cluster_idx]
            
            # Convert token-level parameters to block-level
            window_blocks = config['window'] // self.block_size if config['window'] > 0 else 0
            stride_blocks = max(1, config['stride'] // self.block_size) if config['stride'] > 0 else 0
            
            for i in range(num_blocks):
                # Local window attention
                if window_blocks > 0:
                    start_block = max(0, i - window_blocks)
                    end_block = min(num_blocks, i + window_blocks + 1)
                    layout[start_h:end_h, i, start_block:end_block] = True
                
                # Strided attention
                if stride_blocks > 0:
                    strided_blocks = torch.arange(0, num_blocks, stride_blocks, device=device)
                    layout[start_h:end_h, i, strided_blocks] = True
                
                # Global attention (first, middle, last blocks)
                if config['window'] == 0:
                    global_blocks = torch.tensor([0, num_blocks//2, num_blocks-1], device=device)
                    layout[start_h:end_h, i, global_blocks] = True
        
        return layout

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        attn_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute block-sparse multi-head attention with optimized memory access
        """
        bsz, seq_len, _ = query.size()
        device = query.device

        # 1) Compute Q/K/V with parallel projections
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # 2) Reshape to [B, H, L, D]
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 3) Scale query
        q = q / self.scale

        # 4) Compute attention with memory-efficient implementation
        if XFORMERS_AVAILABLE:
            # Use xformers memory-efficient attention if available
            attn_output = xops.memory_efficient_attention(
                q, k, v,
                attn_bias=xops.LowerTriangularMask() if attn_mask is None else attn_mask,
                p=self.dropout if self.training else 0.0,
                scale=None  # Already scaled q
            )
        else:
            # Get or create block-sparse layout
            cache_key = f"block_layout_{seq_len}"
            if cache_key not in self._mask_cache:
                self._mask_cache[cache_key] = self._create_block_sparse_layout(seq_len, device)
            layout = self._mask_cache[cache_key]

            # Compute attention scores efficiently
            attn_weights = torch.zeros(bsz, self.num_heads, seq_len, seq_len, 
                                    dtype=q.dtype, device=device)
            
            # Only compute attention for non-masked blocks
            block_indices = layout.nonzero(as_tuple=True)
            for h_idx, i, j in zip(*block_indices):
                start_i = i * self.block_size
                end_i = min(start_i + self.block_size, seq_len)
                start_j = j * self.block_size
                end_j = min(start_j + self.block_size, seq_len)
                
                scores = torch.matmul(
                    q[:, h_idx:h_idx+1, start_i:end_i, :],
                    k[:, h_idx:h_idx+1, start_j:end_j, :].transpose(-2, -1)
                )
                attn_weights[:, h_idx:h_idx+1, start_i:end_i, start_j:end_j] = scores

            # Apply attention mask if provided
            if attn_mask is not None:
                attn_weights = attn_weights.masked_fill(~attn_mask.unsqueeze(0), float('-inf'))

            # Apply padding mask if provided
            if attn_padding_mask is not None:
                attn_weights = attn_weights.masked_fill(
                    ~attn_padding_mask.view(bsz, 1, 1, seq_len), float('-inf'))

            # Compute attention probabilities
            attn_probs = F.softmax(attn_weights, dim=-1)
            attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)

            # Apply attention to values
            attn_output = torch.matmul(attn_probs, v)

        # 5) Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.embedding_dim)
        output = self.out_proj(attn_output)

        if need_weights:
            return output, attn_weights if not XFORMERS_AVAILABLE else None
        return output, None

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
            attn_out, attn_weights = self.attention(x2, x2, x2, mask, return_attention=True)
            x = x + self.dropout1(attn_out)
            return x, attn_weights
        else:
            x = x + self.dropout1(self.attention(x2, x2, x2, mask))
        
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