import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import random

class SparseMultiheadAttention(nn.Module):
    """Multihead attention with static sparse patterns based on cluster analysis."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1,
                 max_seq_length: int = 1024, layer_idx: int = 0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.layer_idx = layer_idx
        self.scaling = float(self.head_dim) ** -0.5  # Add explicit scaling factor
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Linear projections with proper initialization
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)  # Added bias
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        
        # Initialize with better scaling for stability
        std = 0.02  # Standard initialization scale
        nn.init.normal_(self.q_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.k_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=std)
        
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.out_proj.bias)
        
        # Sparse attention parameters adjusted for better coverage
        self.local_window = 64  # Increased for better local context
        self.stride = 4  # Reduced stride for better overlap
        self.num_global_tokens = 8  # Increased global tokens
        self.max_seq_length = max_seq_length
        
        # Adjusted head distribution based on empirical analysis
        self.num_local_heads = 4  # Local context (50%)
        self.num_strided_heads = 3  # Medium-range (37.5%)
        self.num_global_heads = 1  # Global context (12.5%)
        
        # Adjusted sparsity levels for better balance
        self.sparsity_levels = {
            'local_low': 0.15,    # Increased local connectivity
            'local_high': 0.25,   # More connections for high-entropy local
            'strided': 0.35,      # Better medium-range coverage
            'global': 0.50        # Reduced sparsity for global attention
        }
        
        # Adjusted target entropy values
        self.target_entropy = {
            'local_low': 2.0,    # Increased from 1.821
            'local_high': 3.0,   # Increased from 2.894
            'strided': 3.5,      # Reduced from 3.800
            'global': 4.0        # Reduced from 4.598
        }
        
        # Cache for efficient computation
        self._mask_cache = {}
        
        # Add layer normalization for stability
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_k = nn.LayerNorm(embed_dim)
        self.norm_v = nn.LayerNorm(embed_dim)
    
    def _compute_entropy(self, mask: torch.Tensor) -> torch.Tensor:
        """Compute attention entropy for the mask."""
        # Convert mask to probabilities with better numerical stability
        mask_float = mask.float()
        row_sums = mask_float.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        probs = mask_float / row_sums
        
        # Compute entropy with numerical stability
        eps = 1e-8
        entropy = -(probs * (probs + eps).log()).sum(dim=-1).mean()
        return entropy
    
    def _adjust_mask_for_entropy(self, mask: torch.Tensor, target_entropy: float, 
                               max_iterations: int = 5) -> torch.Tensor:
        """Adjust mask to match target entropy with improved stability."""
        current_entropy = self._compute_entropy(mask)
        
        # Binary search with better convergence
        left, right = 0.0, 1.0
        best_mask = mask.clone()
        best_entropy_diff = abs(current_entropy - target_entropy)
        
        for _ in range(max_iterations):
            if best_entropy_diff < 0.05:  # Tighter convergence threshold
                break
                
            sparsity = (left + right) / 2
            temp_mask = mask.clone()
            
            # Compute number of connections based on sparsity
            num_tokens = mask.size(-1)
            target_connections = int(num_tokens * (1 - sparsity))
            
            # Process each row independently
            for i in range(mask.size(0)):
                # Get current connections
                curr_connections = temp_mask[i].sum().item()
                
                if curr_connections < target_connections:
                    # Need to add connections
                    zero_indices = torch.where(~temp_mask[i])[0]
                    if zero_indices.numel() > 0:
                        num_to_add = min(target_connections - curr_connections, zero_indices.numel())
                        perm = torch.randperm(zero_indices.numel(), device=mask.device)
                        to_add = zero_indices[perm[:num_to_add]]
                        temp_mask[i, to_add] = True
                else:
                    # Need to remove connections
                    one_indices = torch.where(temp_mask[i])[0]
                    if one_indices.numel() > target_connections:
                        num_to_keep = target_connections
                        perm = torch.randperm(one_indices.numel(), device=mask.device)
                        to_keep = one_indices[perm[:num_to_keep]]
                        temp_mask[i] = False
                        temp_mask[i, to_keep] = True
            
            new_entropy = self._compute_entropy(temp_mask)
            entropy_diff = abs(new_entropy - target_entropy)
            
            if entropy_diff < best_entropy_diff:
                best_mask = temp_mask.clone()
                best_entropy_diff = entropy_diff
            
            if new_entropy < target_entropy:
                right = sparsity
            else:
                left = sparsity
        
        return best_mask
    
    def _create_local_mask(self, seq_length: int) -> torch.Tensor:
        """Create local attention mask with improved coverage."""
        mask = torch.zeros(seq_length, seq_length, dtype=torch.bool)
        
        # Create initial local window with overlap
        window_size = self.local_window
        overlap = window_size // 4  # 25% overlap between windows
        
        for i in range(seq_length):
            # Center window around current position
            center = i
            start = max(0, center - window_size // 2)
            end = min(seq_length, center + window_size // 2 + 1)
            
            # Add main window
            mask[i, start:end] = True
            
            # Add overlapping connections
            if i >= overlap:
                mask[i, i-overlap:i] = True
            if i < seq_length - overlap:
                mask[i, i:i+overlap] = True
        
        # Adjust for target entropy
        head_idx = self.layer_idx % self.num_local_heads
        target_entropy = self.target_entropy['local_low'] if head_idx < self.num_local_heads // 2 \
                        else self.target_entropy['local_high']
        
        mask = self._adjust_mask_for_entropy(mask, target_entropy)
        return mask
    
    def _create_strided_mask(self, seq_length: int) -> torch.Tensor:
        """Create strided attention mask with better coverage."""
        mask = torch.zeros(seq_length, seq_length, dtype=torch.bool)
        
        # Improved strided pattern with multiple scales
        strides = [self.stride, self.stride * 2, self.stride * 4]
        weights = [0.5, 0.3, 0.2]  # Prioritize smaller strides
        
        for i in range(seq_length):
            # Add local context
            start = max(0, i - self.local_window // 4)
            end = min(seq_length, i + self.local_window // 4 + 1)
            mask[i, start:end] = True
            
            # Add multi-scale strided connections
            for stride, weight in zip(strides, weights):
                if random.random() < weight:
                    indices = torch.arange(i % stride, seq_length, stride)
                    mask[i, indices] = True
        
        # Adjust for target entropy
        mask = self._adjust_mask_for_entropy(mask, self.target_entropy['strided'])
        return mask
    
    def _create_global_mask(self, seq_length: int) -> torch.Tensor:
        """Create global attention mask with improved connectivity."""
        mask = torch.zeros(seq_length, seq_length, dtype=torch.bool)
        
        # Improved global token selection
        num_global = self.num_global_tokens
        
        # Always include start, end, and evenly spaced tokens
        global_indices = [0, seq_length - 1]
        if num_global > 2:
            step = seq_length // (num_global - 2)
            for i in range(step, seq_length - 1, step):
                global_indices.append(i)
        
        # Add some random global tokens for diversity
        num_random = max(0, num_global - len(global_indices))
        if num_random > 0:
            available = list(set(range(seq_length)) - set(global_indices))
            random_indices = random.sample(available, min(num_random, len(available)))
            global_indices.extend(random_indices)
        
        global_indices = sorted(list(set(global_indices)))[:num_global]
        
        # Connect global tokens bidirectionally
        mask[:, global_indices] = True
        mask[global_indices, :] = True
        
        # Add some random connections for each token
        num_random = seq_length // 32  # Reduced from previous value
        for i in range(seq_length):
            if i not in global_indices:
                available = list(set(range(seq_length)) - set(global_indices) - {i})
                if available:
                    random_indices = random.sample(available, min(num_random, len(available)))
                    mask[i, random_indices] = True
        
        # Adjust for target entropy
        mask = self._adjust_mask_for_entropy(mask, self.target_entropy['global'])
        return mask
    
    def _get_mask_for_head(self, head_idx: int, seq_length: int) -> torch.Tensor:
        """Get appropriate mask for a specific attention head based on cluster analysis."""
        cache_key = f"{seq_length}_{head_idx}"
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]
        
        # Assign masks based on head index and cluster distribution
        if head_idx < self.num_local_heads:  # Clusters 0 & 3 (focused)
            mask = self._create_local_mask(seq_length)
        elif head_idx < self.num_local_heads + self.num_strided_heads:  # Cluster 1 (diffuse)
            mask = self._create_strided_mask(seq_length)
        else:  # Cluster 2 (most diffuse)
            mask = self._create_global_mask(seq_length)
        
        # Always allow self-attention
        mask.fill_diagonal_(True)
        
        # Cache the mask
        self._mask_cache[cache_key] = mask
        return mask
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with improved attention computation."""
        batch_size, seq_length, _ = query.shape
        
        # Apply layer normalization for stability
        query = self.norm_q(query)
        key = self.norm_k(key)
        value = self.norm_v(value)
        
        # Linear projections with scaled dot-product attention
        q = self.q_proj(query).view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, seq_length, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, seq_length, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch_size, num_heads, seq_length, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Apply scaling factor
        q = q * self.scaling
        
        # Compute attention scores with improved numerical stability
        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        
        # Apply sparse attention patterns
        for head_idx in range(self.num_heads):
            head_mask = self._get_mask_for_head(head_idx, seq_length).to(query.device)
            attn_weights[:, head_idx] = attn_weights[:, head_idx].masked_fill(~head_mask, float('-inf'))
        
        # Apply causal mask if provided
        if attn_mask is not None:
            attn_weights = attn_weights.masked_fill(attn_mask.unsqueeze(1), float('-inf'))
        
        # Apply key padding mask if provided
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        # Compute attention probabilities with improved numerical stability
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights.type_as(value)
        
        # Apply dropout
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # Compute output
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_dim)
        output = self.out_proj(output)
        
        return output, attn_weights

class SparseTransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with sparse attention."""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "gelu", layer_idx: int = 0):
        super().__init__()
        
        self.self_attn = SparseMultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            layer_idx=layer_idx
        )
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model, eps=1e-4)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-4)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = F.gelu if activation == "gelu" else F.relu
        
        # Initialize weights
        nn.init.xavier_uniform_(self.linear1.weight, gain=0.1)
        nn.init.xavier_uniform_(self.linear2.weight, gain=0.1)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)
    
    def forward(self, src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Pass the input through the encoder layer.
        
        Args:
            src: Source tensor [batch_size, seq_length, d_model]
            src_mask: Optional mask [seq_length, seq_length]
            src_key_padding_mask: Optional mask [batch_size, seq_length]
            
        Returns:
            Output tensor of shape [batch_size, seq_length, d_model]
        """
        # Self attention
        src2 = self.norm1(src)
        src2, _ = self.self_attn(
            src2, src2, src2,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(src2)
        
        # Feed-forward
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        
        return src

class SparseTransformer(nn.Module):
    """Transformer model with sparse attention patterns."""
    
    def __init__(self, vocab_size: int, d_model: int = 512, nhead: int = 8,
                 num_layers: int = 6, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "gelu",
                 max_seq_length: int = 1024):
        super().__init__()
        
        self.model_type = 'Transformer'
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_seq_length = max_seq_length
        
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Create transformer layers with exact naming to match checkpoint
        self.layers = nn.ModuleList([
            SparseTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                layer_idx=i
            ) for i in range(num_layers)
        ])
        
        # Use exact names from checkpoint
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters."""
        # Initialize embedding
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        
        # Initialize output projection
        nn.init.normal_(self.fc_out.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.fc_out.bias)
    
    def forward(self, src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len]
            src_mask: Optional tensor, shape [seq_len, seq_len]
            src_key_padding_mask: Optional tensor, shape [batch_size, seq_len]
            
        Returns:
            output Tensor of shape [batch_size, seq_len, vocab_size]
        """
        # Ensure input sequence length doesn't exceed maximum
        if src.size(1) > self.max_seq_length:
            src = src[:, :self.max_seq_length]
            if src_mask is not None:
                src_mask = src_mask[:self.max_seq_length, :self.max_seq_length]
            if src_key_padding_mask is not None:
                src_key_padding_mask = src_key_padding_mask[:, :self.max_seq_length]
        
        # Embedding and positional encoding
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        
        # Final layer norm and output projection
        x = self.norm(x)
        output = self.fc_out(x)
        
        return output

class PositionalEncoding(nn.Module):
    """Positional encoding for the transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.
        
        Args:
            x: Input tensor [batch_size, seq_length, d_model]
            
        Returns:
            Output tensor of same shape with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x) 