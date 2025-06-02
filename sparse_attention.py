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
    Multi‐head attention with four distinct sparse patterns (clusters).
    Cluster 0: narrow local window ±8
    Cluster 1: local window ±16 + strided every 8th
    Cluster 2: no local window, only global anchors {0, L//2, L-1} + strided every 32nd
    Cluster 3: wide local window ±16
    """

    def __init__(self, embedding_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads     = num_heads
        self.dropout       = dropout
        self.head_dim      = embedding_dim // num_heads

        assert self.head_dim * num_heads == embedding_dim, \
            "embedding_dim must be divisible by num_heads"
        assert num_heads % 4 == 0, \
            "num_heads must be divisible by 4 (we divide heads into 4 clusters)"

        self.heads_per_cluster = num_heads // 4

        # Q/K/V projections + final output projection
        self.q_proj   = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.k_proj   = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.v_proj   = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)

        self.scale = math.sqrt(self.head_dim)

        # Cache masks directly on GPU to avoid CPU->GPU transfer
        self._mask_cache = {}
        
        # Pre-compute cluster configurations
        self.cluster_configs = {
            0: {'window': 8,  'stride': 1,  'global': False},  # narrow local
            1: {'window': 32, 'stride': 8,  'global': False},  # local + strided
            2: {'window': 0,  'stride': 32, 'global': True},   # global + strided
            3: {'window': 32, 'stride': 1,  'global': False}   # wide local
        }

    def _shape(self, tensor, seq_len, bsz):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _create_sparse_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create all sparse attention masks at once for better parallelization.
        Returns a tensor of shape [num_heads, seq_len, seq_len]
        """
        # Initialize full mask for all heads
        full_mask = torch.zeros(self.num_heads, seq_len, seq_len, 
                              dtype=torch.bool, device=device)
        
        # Create masks for all heads in parallel
        pos = torch.arange(seq_len, device=device)
        
        # Process each cluster type
        heads_per_cluster = self.num_heads // 4
        for cluster_idx in range(4):
            start_h = cluster_idx * heads_per_cluster
            end_h = (cluster_idx + 1) * heads_per_cluster
            
            if cluster_idx == 0:  # Narrow local window (±8)
                window = 8
                dist = pos.unsqueeze(1) - pos.unsqueeze(0)
                window_mask = (dist.abs() <= window // 2)
                full_mask[start_h:end_h] |= window_mask
                
            elif cluster_idx == 1:  # Local + strided (±16 + every 8th)
                window = 32
                dist = pos.unsqueeze(1) - pos.unsqueeze(0)
                window_mask = (dist.abs() <= window // 2)
                stride_pos = torch.arange(0, seq_len, 8, device=device)
                stride_mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
                stride_mask[:, stride_pos] = True
                full_mask[start_h:end_h] |= (window_mask | stride_mask)
                
            elif cluster_idx == 2:  # Global anchors + strided (every 32nd)
                global_pos = torch.tensor([0, seq_len//2, seq_len-1], device=device)
                global_mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
                global_mask[:, global_pos] = True
                stride_pos = torch.arange(0, seq_len, 32, device=device)
                stride_mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
                stride_mask[:, stride_pos] = True
                full_mask[start_h:end_h] |= (global_mask | stride_mask)
                
            else:  # Wide local window (±16)
                window = 32
                dist = pos.unsqueeze(1) - pos.unsqueeze(0)
                window_mask = (dist.abs() <= window // 2)
                full_mask[start_h:end_h] |= window_mask
        
        # Apply causal masking
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
        full_mask &= ~causal_mask.unsqueeze(0)
        
        return full_mask

    def forward(self, query, key, value, attn_mask=None, need_weights=False, attn_padding_mask=None):
        """
        Optimized sparse multi-head attention implementation.
        """
        bsz, seq_len, _ = query.size()
        device = query.device

        # 1) Compute Q/K/V with parallel projections
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # 2) Reshape to [B, H, L, D] in parallel
        q = self._shape(q, seq_len, bsz)
        k = self._shape(k, seq_len, bsz)
        v = self._shape(v, seq_len, bsz)

        # 3) Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # 4) Apply sparse attention masks (cached and parallelized)
        cache_key = f"sparse_mask_{seq_len}"
        if cache_key not in self._mask_cache:
            self._mask_cache[cache_key] = self._create_sparse_mask(seq_len, device)
        
        sparse_mask = self._mask_cache[cache_key]
        fill_val = -65504.0 if attn_scores.dtype == torch.float16 else -1e9
        attn_scores = attn_scores.masked_fill(~sparse_mask.unsqueeze(0), fill_val)

        # 5) Apply attention mask if provided
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
            attn_scores = attn_scores.masked_fill(~attn_mask.unsqueeze(1), fill_val)

        # 6) Apply padding mask if provided
        if attn_padding_mask is not None:
            pad_mask = ~attn_padding_mask.view(bsz, 1, 1, seq_len)
            attn_scores = attn_scores.masked_fill(pad_mask, fill_val)

        # 7) Compute attention probabilities with optimized memory access
        attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32)
        if attn_scores.dtype == torch.float16:
            attn_probs = attn_probs.to(torch.float16)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)

        # 8) Compute output with parallel matrix multiplication
        context = torch.matmul(attn_probs, v)
        context = context.transpose(1, 2).reshape(bsz, seq_len, self.embedding_dim)
        output = self.out_proj(context)

        if need_weights:
            return output, attn_probs

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