"""
Optimized sparse attention implementation with block-sparse patterns and efficient memory usage.
Uses block-sparse attention with efficient CUDA operations.
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
    warnings.warn("xformers not available. Using optimized block-sparse implementation.")

class BlockSparseAttention:
    """Helper class for efficient block-sparse attention computation"""
    def __init__(self, block_size: int = 32):
        self.block_size = block_size
        self.layout_cache = {}
    
    @staticmethod
    @torch.jit.script
    def _compute_block_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """JIT-optimized block attention computation"""
        scores = torch.matmul(q, k.transpose(-2, -1))
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        attn_probs = F.softmax(scores, dim=-1)
        return torch.matmul(attn_probs, v)

    @torch.jit.script
    def _process_block(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                      start_i: int, end_i: int, start_j: int, end_j: int) -> torch.Tensor:
        """Process a single attention block with optimized memory access"""
        q_block = q[..., start_i:end_i, :]
        k_block = k[..., start_j:end_j, :]
        v_block = v[..., start_j:end_j, :]
        return self._compute_block_attention(q_block, k_block, v_block)

class SparseMultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embedding_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        assert self.head_dim * num_heads == embedding_dim, "embedding_dim must be divisible by num_heads"
        assert num_heads % 4 == 0, "num_heads must be divisible by 4 for cluster patterns"

        # Q/K/V projections for all heads at once
        self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)

        # Initialize block-sparse helper
        self.block_sparse = BlockSparseAttention()
        
        # Initialize attention patterns
        self.heads_per_pattern = num_heads // 4
        self._setup_attention_patterns()

    def _setup_attention_patterns(self):
        """Setup optimized attention patterns"""
        self.patterns = {
            'narrow_local': {'window': 8, 'stride': 0},     # ±8 local window
            'wide_strided': {'window': 32, 'stride': 8},    # ±16 + strided/8
            'global_strided': {'window': 0, 'stride': 32},  # Global + strided/32
            'wide_local': {'window': 32, 'stride': 0}       # ±16 local window
        }

    @torch.jit.script
    def _create_pattern_mask(self, seq_len: int, window: int, stride: int, 
                           is_global: bool, device: torch.device) -> torch.Tensor:
        """JIT-optimized pattern mask creation"""
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
        
        if window > 0:
            positions = torch.arange(seq_len, device=device)
            dist = positions.unsqueeze(1) - positions.unsqueeze(0)
            mask |= dist.abs() <= window // 2
            
        if stride > 0:
            stride_positions = torch.arange(0, seq_len, stride, device=device)
            mask[:, stride_positions] = True
            
        if is_global:
            global_positions = torch.tensor([0, seq_len//2, seq_len-1], device=device)
            mask[:, global_positions] = True
            
        return mask

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
               attn_mask: Optional[torch.Tensor] = None,
               need_weights: bool = False,
               attn_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Efficient block-sparse attention with optimized memory access and computation
        """
        bsz, seq_len, _ = query.size()
        device = query.device

        # 1) Project and reshape Q/K/V
        q = self.q_proj(query).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2) / self.scale
        k = self.k_proj(key).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if XFORMERS_AVAILABLE:
            # Use xformers if available
            attn_output = xops.memory_efficient_attention(
                q, k, v,
                attn_bias=xops.LowerTriangularMask() if attn_mask is None else attn_mask,
                p=self.dropout if self.training else 0.0
            )
        else:
            # Efficient block-sparse implementation
            attn_output = torch.zeros_like(q)
            
            # Process each attention pattern
            for pattern_idx, (pattern_name, pattern) in enumerate(self.patterns.items()):
                start_head = pattern_idx * self.heads_per_pattern
                end_head = (pattern_idx + 1) * self.heads_per_pattern
                
                # Create or get cached pattern mask
                cache_key = f"{pattern_name}_{seq_len}"
                if cache_key not in self.block_sparse.layout_cache:
                    mask = self._create_pattern_mask(
                        seq_len, pattern['window'], pattern['stride'],
                        pattern_name == 'global_strided', device
                    )
                    self.block_sparse.layout_cache[cache_key] = mask
                
                pattern_mask = self.block_sparse.layout_cache[cache_key]
                
                # Process blocks for this pattern
                for i in range(0, seq_len, self.block_sparse.block_size):
                    end_i = min(i + self.block_sparse.block_size, seq_len)
                    for j in range(0, seq_len, self.block_sparse.block_size):
                        end_j = min(j + self.block_sparse.block_size, seq_len)
                        
                        if pattern_mask[i:end_i, j:end_j].any():
                            block_output = self.block_sparse._process_block(
                                q[:, start_head:end_head],
                                k[:, start_head:end_head],
                                v[:, start_head:end_head],
                                i, end_i, j, end_j
                            )
                            attn_output[:, start_head:end_head, i:end_i] = block_output

            # Apply dropout
            if self.training and self.dropout > 0:
                attn_output = F.dropout(attn_output, p=self.dropout, training=True)

        # Final output projection
        output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.embedding_dim)
        output = self.out_proj(output)

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