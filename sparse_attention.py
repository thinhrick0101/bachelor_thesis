import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class SparseAttention(nn.Module):
    """
    Sparse Attention mechanism with multiple sparsity patterns based on analysis:
    - Early layers (0-3): More dense attention with local focus
    - Middle layers (4-8): Mixed local and global attention
    - Later layers (9+): Highly sparse attention with selective global connections
    """
    def __init__(self, embed_dim, num_heads, layer_idx, dropout=0.1, 
                 local_window=32, global_tokens=8, sparsity_factor=0.2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.layer_idx = layer_idx
        
        # Sparsity hyperparameters
        self.local_window = local_window
        self.global_tokens = global_tokens
        self.sparsity_factor = sparsity_factor
        
        # Learnable temperature parameter per head
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        # Linear transformations
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Initialize with better scaling
        self._reset_parameters()
        
        # Cache for attention masks
        self._mask_cache = {}

    def _reset_parameters(self):
        # Initialize with scaled Xavier uniform
        gain = math.sqrt(2.0)  # ReLU gain
        nn.init.xavier_uniform_(self.q_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=gain)
        
        # Initialize biases to zero
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.zeros_(self.out_proj.bias)
        
        # Initialize temperature with slight noise
        with torch.no_grad():
            self.temperature.data.fill_(1.0)
            self.temperature.data += torch.randn_like(self.temperature.data) * 0.01

    def _create_sparse_mask(self, seq_len, device):
        """Create sparse attention mask based on layer position and analysis patterns"""
        cache_key = f"{seq_len}_{self.layer_idx}"
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key].to(device)
        
        # Initialize mask
        mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)
        
        # Early layers (0-3): Dense local attention with gradual sparsification
        if self.layer_idx < 4:
            local_size = min(self.local_window * 2, seq_len)
            for i in range(seq_len):
                start = max(0, i - local_size // 2)
                end = min(seq_len, i + local_size // 2 + 1)
                mask[i, start:end] = True
                
            # Add some global connections
            if self.layer_idx > 0:
                stride = max(1, seq_len // (self.global_tokens * (4 - self.layer_idx)))
                global_idx = torch.arange(0, seq_len, stride, device=device)
                mask[:, global_idx] = True
        
        # Middle layers (4-8): Mixed local and global attention
        elif self.layer_idx < 9:
            # Local window attention
            for i in range(seq_len):
                start = max(0, i - self.local_window // 2)
                end = min(seq_len, i + self.local_window // 2 + 1)
                mask[i, start:end] = True
            
            # Global tokens with adaptive stride
            stride = max(1, seq_len // (self.global_tokens * 2))
            global_idx = torch.arange(0, seq_len, stride, device=device)
            mask[:, global_idx] = True
            
            # Add periodic attention
            period = max(1, seq_len // 16)
            for i in range(seq_len):
                mask[i, i::period] = True
        
        # Later layers (9+): Highly selective sparse attention
        else:
            # Reduced local window
            local_size = self.local_window // 2
            for i in range(seq_len):
                start = max(0, i - local_size // 2)
                end = min(seq_len, i + local_size // 2 + 1)
                mask[i, start:end] = True
            
            # Selective global tokens
            num_global = min(self.global_tokens, seq_len // 8)
            stride = max(1, seq_len // num_global)
            global_idx = torch.arange(0, seq_len, stride, device=device)
            mask[:, global_idx] = True
            
            # Add learned important positions based on entropy analysis
            sparsity_threshold = self.sparsity_factor * (1 + self.layer_idx / 12)
            num_extra = int(seq_len * sparsity_threshold)
            if num_extra > 0:
                for i in range(seq_len):
                    # Random but consistent extra connections
                    rand_idx = torch.randperm(seq_len, device=device)[:num_extra]
                    mask[i, rand_idx] = True
        
        # Always attend to self
        mask.fill_diagonal_(True)
        
        # Cache the mask
        self._mask_cache[cache_key] = mask.cpu()
        return mask

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        """
        Forward pass with sparse attention masking
        query, key, value: (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = query.shape
        scaling = self.scaling
        
        # Linear transformations and reshape
        q = self.q_proj(query).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scaling  # [batch, heads, seq_len, seq_len]
        
        # Apply learned temperature scaling
        attn_weights = attn_weights * F.softplus(self.temperature)
        
        # Create and apply sparse mask
        sparse_mask = self._create_sparse_mask(seq_len, query.device)
        sparse_mask = sparse_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        attn_weights = attn_weights.masked_fill(~sparse_mask, float('-inf'))
        
        # Apply additional masks if provided
        if attn_mask is not None:
            # Ensure attn_mask has correct shape [batch_size, seq_len, seq_len] or [seq_len, seq_len]
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)  # Add batch dimension if needed
            
            # Add head dimension and convert to float
            attn_mask = attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1).to(dtype=torch.float32)
            attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask == 1, 0.0)
            attn_weights = attn_weights + attn_mask
            
        if key_padding_mask is not None:
            # key_padding_mask should be [batch_size, seq_len]
            # Convert to attention mask shape [batch_size, 1, 1, seq_len]
            key_padding_mask = key_padding_mask.float().unsqueeze(1).unsqueeze(2)
            key_padding_mask = key_padding_mask.expand(-1, self.num_heads, seq_len, -1)
            key_padding_mask = key_padding_mask.masked_fill(key_padding_mask == 0, float('-inf'))
            attn_weights = attn_weights + key_padding_mask
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # Compute output
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        output = self.out_proj(output)
        
        if need_weights:
            return output, attn_weights
        return output, None

class AdaptiveSparseAttention(nn.Module):
    """
    Adaptive sparse attention that learns sparsity patterns during training
    """
    def __init__(self, embed_dim, num_heads, layer_idx, dropout=0.1):
        super().__init__()
        self.base_attention = SparseAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            layer_idx=layer_idx,
            dropout=dropout
        )
        
        # Learnable sparsity parameters
        self.sparsity_control = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.content_importance = nn.Linear(embed_dim, num_heads)
        
    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        # Get content-based importance scores
        importance = torch.sigmoid(self.content_importance(query))
        importance = importance.transpose(1, 2).unsqueeze(-1)  # [batch, heads, seq_len, 1]
        
        # Modify the sparsity pattern based on content
        sparsity_factor = torch.sigmoid(self.sparsity_control) * importance
        
        # Apply base attention with adaptive sparsity
        self.base_attention.sparsity_factor = sparsity_factor.mean().item()
        return self.base_attention(query, key, value, key_padding_mask, need_weights, attn_mask) 