"""
Sparse attention implementation with efficient masking and caching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SparseMultiHeadAttention(nn.Module):
    """
    Multi‐head attention with aggressive sparse patterns to speed up learning:
    - Cluster 0: focused local (±16) for essential local context
    - Cluster 3: diffuse attention (±32, stride=4) for medium-range
    - Cluster 1: strided attention (±64, stride=8) for long-range
    - Cluster 2: global attention with dense anchor points
    """

    def __init__(self, embedding_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads     = num_heads
        self.dropout       = dropout
        self.head_dim      = embedding_dim // num_heads

        assert self.head_dim * num_heads == embedding_dim, \
            "embedding_dim must be divisible by num_heads"
        assert num_heads == 8, \
            "This implementation assumes 8 heads for optimal distribution"

        # More aggressive head distribution favoring global/diffuse attention
        self.cluster_head_counts = {
            0: 1,  # 1 head focused local (reduced from 2)
            3: 2,  # 2 heads diffuse medium-range (reduced from 4)
            1: 3,  # 3 heads strided long-range (increased from 1)
            2: 2   # 2 heads global (increased from 1)
        }

        # Q/K/V projections + final output projection
        self.q_proj   = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.k_proj   = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.v_proj   = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)

        self.scale = math.sqrt(self.head_dim)

        # Cache for CPU masks
        self._mask_cache = {}

        # Increased window sizes for better coverage
        self.window_sizes = {
            0: 16,    # Local: increased to ±16 for better local context
            3: 32,    # Medium: increased to ±32 for diffuse coverage
            1: 64,    # Long-range: increased to ±64 for broader context
            2: 128    # Global: maximum context with anchor points
        }

        # More aggressive stride patterns
        self.strides = {
            0: 1,     # No stride for focused local
            3: 4,     # Small stride for diffuse (reduced from 8)
            1: 8,     # Medium stride for long-range
            2: 16     # Large stride for global coverage
        }

        # Denser global anchor points for faster information flow
        self.global_anchors = [
            0.0,      # Start
            0.1,      # Early context
            0.2,      # Early-mid
            0.3,      # Early-mid
            0.382,    # Golden ratio point
            0.5,      # Middle
            0.618,    # Inverse golden ratio
            0.7,      # Late-mid
            0.8,      # Late-mid
            0.9,      # Late context
            1.0       # End
        ]

    def _shape(self, tensor, seq_len, bsz):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _create_cluster_mask(self, seq_len, cluster_idx):
        """
        Build attention mask for each cluster:
        - Clusters 0,3: Pure local attention with different windows
        - Cluster 1: Strided attention for efficient long-range coverage
        - Cluster 2: Global attention with strategic anchor points
        """
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)

        window_size = self.window_sizes[cluster_idx]
        stride = self.strides[cluster_idx]

        for i in range(seq_len):
            # 1) Local window with appropriate size
            if window_size > 0:
                half = window_size // 2
                start = max(0, i - half)
                end = min(seq_len, i + half + 1)
                mask[i, start:end] = True

            # 2) Strided attention (for clusters 1 and 2)
            if stride > 1:
                strided_indices = torch.arange(0, seq_len, stride)
                mask[i, strided_indices] = True

            # 3) Global anchors (only for cluster 2)
            if cluster_idx == 2:
                for ratio in self.global_anchors:
                    idx = min(seq_len - 1, int(ratio * seq_len))
                    mask[i, idx] = True

        # Make causal (lower triangular)
        mask = mask.tril()
        return mask

    def forward(self, query, key, value, attn_mask=None, need_weights=False, attn_padding_mask=None):
        """
        Standard multi-head attention interface with empirically optimized head distribution.
        
        Args:
            query: [B, L, E]
            key: [B, L, E]  
            value: [B, L, E]
            attn_mask: Combined causal+padding mask [B, L, L] or [L, L]
            need_weights: If True, return attention probabilities
            attn_padding_mask: Optional padding mask [B, L]
        
        Returns:
            if need_weights=False:  → [B, L, E]
            if need_weights=True:   → ([B, L, E], [B, H, L, L])
        """
        bsz, seq_len, _ = query.size()
        device = query.device

        # 1) compute Q/K/V
        q = self.q_proj(query)  # [B, L, E]
        k = self.k_proj(key)
        v = self.v_proj(value)

        # 2) reshape → [B, H, L, head_dim]
        q = self._shape(q, seq_len, bsz)
        k = self._shape(k, seq_len, bsz)
        v = self._shape(v, seq_len, bsz)

        # 3) raw dot‐product scores: [B, H, L, L]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # 4) apply cluster masks head‐by‐head using new distribution
        current_head = 0
        for cluster_idx in [0, 3, 1, 2]:  # Order by entropy: focused → global
            num_heads = self.cluster_head_counts[cluster_idx]
            end_head = current_head + num_heads
            
            cache_key = (cluster_idx, seq_len)
            if cache_key not in self._mask_cache:
                cpu_mask = self._create_cluster_mask(seq_len, cluster_idx)
                self._mask_cache[cache_key] = cpu_mask

            cluster_mask = self._mask_cache[cache_key].to(device)
            inv = ~cluster_mask

            fill_val = -65504.0 if attn_scores.dtype == torch.float16 else -1e9
            attn_scores[:, current_head:end_head] = attn_scores[:, current_head:end_head].masked_fill(
                inv.unsqueeze(0).unsqueeze(0),
                fill_val
            )
            
            current_head = end_head

        # 5) apply attention mask if provided
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
            
            fill_val = -65504.0 if attn_scores.dtype == torch.float16 else -1e9
            attn_scores = attn_scores.masked_fill(~attn_mask.unsqueeze(1), fill_val)

        # 6) apply padding mask if provided
        if attn_padding_mask is not None:
            pad_mask = ~attn_padding_mask.view(bsz, 1, 1, seq_len)
            fill_val = -65504.0 if attn_scores.dtype == torch.float16 else -1e9
            attn_scores = attn_scores.masked_fill(pad_mask, fill_val)

        # 7) softmax + dropout → attention_probs
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)

        # 8) weighted sum → [B, H, L, head_dim]
        context = torch.matmul(attn_probs, v)

        # 9) restore → [B, L, E]
        context = context.transpose(1, 2).reshape(bsz, seq_len, self.embedding_dim)

        # 10) final linear
        out = self.out_proj(context)

        if need_weights:
            return out, attn_probs

        return out


