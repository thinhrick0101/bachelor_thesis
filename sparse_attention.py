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
    Cluster 1: local window ±16 + strided every 8th
    Cluster 2: no local window, only global anchors {0, L//2, L-1} + strided every 32nd
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

        # We will cache CPU masks in self._mask_cache[ (cluster_idx, seq_len) ] = boolean_tensor [L, L]
        self._mask_cache = {}

        # Exactly match the “±8 / ±16 / global / ±16” spec:
        #   cluster 0: ±8 → window_size=16
        #   cluster 1: ±16 → window_size=32, plus stride=8
        #   cluster 2: window_size=0 (no local), stride=32, plus anchors
        #   cluster 3: ±16 → window_size=32, stride=1
        self.window_sizes = {
            0: 8,   # cluster 0 = ±8
            1: 32,   # cluster 1 = ±16
            2: 64,    # cluster 2 = no local window
            3: 32    # cluster 3 = ±16
        }
        self.strides = {
            0: 1,    # cluster 0: no stride
            1: 8,    # cluster 1: stride every 8th
            2: 32,   # cluster 2: stride every 32nd
            3: 1     # cluster 3: no stride
        }

    def _shape(self, tensor, seq_len, bsz):
        # After linear, we have [B, L, E]. 
        # Reshape → [B, L, heads, head_dim], then transpose → [B, heads, L, head_dim].
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _create_cluster_mask(self, seq_len, cluster_idx):
        """
        Build a CPU‐side boolean mask [L, L] for cluster_idx.
        True means “allowed to attend,” False means “mask out.”
        """
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)  # on CPU

        window_size = self.window_sizes[cluster_idx]
        stride      = self.strides[cluster_idx]

        for i in range(seq_len):
            # 1) local window if window_size > 0
            if window_size > 0:
                half = window_size // 2
                start = max(0, i - half)
                end   = min(seq_len, i + half + 1)
                mask[i, start:end] = True

            # 2) strided attention if stride > 1
            if stride > 1:
                strided_indices = torch.arange(0, seq_len, stride)
                mask[i, strided_indices] = True

            # 3) cluster 2 global anchors only
            if cluster_idx == 2:
                # always attend to first token (0) and last token (L-1)
                mask[i, 0]       = True
                mask[i, seq_len-1] = True
                # attend to the middle token
                mid = seq_len // 2
                mask[i, mid] = True
        mask = mask.tril()
        return mask  # CPU boolean tensor [L, L]

    def forward(self, x, attn_padding_mask=None, return_attention=False):
        """
        x:                    [B, L, E]
        attn_padding_mask:    (optional) boolean [B, L]: True for real tokens, False for pad.
                              We will combine this with the causal mask in the caller.
        return_attention:     if True, return (output, attention_probs).

        returns:
          if return_attention=False:  → [B, L, E]
          if return_attention=True:   → ( [B, L, E], [B, H, L, L] ) 
        """
        bsz, seq_len, _ = x.size()
        device = x.device

        # 1) compute Q/K/V
        q = self.q_proj(x)  # [B, L, E]
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2) reshape → [B, H, L, head_dim]
        q = self._shape(q, seq_len, bsz)
        k = self._shape(k, seq_len, bsz)
        v = self._shape(v, seq_len, bsz)

        # 3) raw dot‐product scores: [B, H, L, L]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # 4) apply cluster masks head‐by‐head
        for cluster_idx in range(4):
            start_h = cluster_idx * self.heads_per_cluster
            end_h   = (cluster_idx + 1) * self.heads_per_cluster
            cache_key = (cluster_idx, seq_len)

            if cache_key not in self._mask_cache:
                # build on CPU once
                cpu_mask = self._create_cluster_mask(seq_len, cluster_idx)  # [L, L] on CPU
                self._mask_cache[cache_key] = cpu_mask

            # move to GPU if needed
            cluster_mask = self._mask_cache[cache_key].to(device)  # [L, L]

            # invert to mask out
            inv = ~cluster_mask  # [L, L] bool

            # choose large negative fill
            if attn_scores.dtype == torch.float16:
                fill_val = -65504.0
            else:
                fill_val = -1e9

            attn_scores[:, start_h:end_h] = attn_scores[:, start_h:end_h].masked_fill(
                inv.unsqueeze(0).unsqueeze(0),  # → [1, 1, L, L]
                fill_val
            )

        # 5) if caller passed a padding‐mask [B, L], incorporate it now.
        #    We assume the caller built a combined causal+padding mask of shape [B, L, L].
        if attn_padding_mask is not None:
            # attn_padding_mask: [B, L], True = real token, False = pad
            # We need shape [B, 1, 1, L], so that for each head & query i,
            # we block out key=j if attn_padding_mask[b,j] == False.
            pad_mask = ~attn_padding_mask.view(bsz, 1, 1, seq_len)
            if attn_scores.dtype == torch.float16:
                fill_val = -65504.0
            else:
                fill_val = -1e9
            attn_scores = attn_scores.masked_fill(pad_mask, fill_val)

        # 6) softmax + dropout → attention_probs
        attn_probs = F.softmax(attn_scores, dim=-1)  # [B, H, L, L]
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)

        # 7) weighted sum → [B, H, L, head_dim]
        context = torch.matmul(attn_probs, v)

        # 8) restore → [B, L, E]
        context = context.transpose(1, 2).reshape(bsz, seq_len, self.embedding_dim)

        # 9) final linear
        out = self.out_proj(context)  # [B, L, E]

        if return_attention:
            return out, attn_probs

        return out



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