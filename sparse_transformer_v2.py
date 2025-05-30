import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from torch.utils.checkpoint import checkpoint

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
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Initialize with smaller weights for better gradient flow
        nn.init.xavier_uniform_(self.q_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.1)
        
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.out_proj.bias)
        
        # Sparse attention parameters adjusted based on cluster analysis
        # Cluster 0 & 3: More local heads (focused, uniform)
        # Cluster 1: Medium sparsity heads (diffuse, sparse)
        # Cluster 2: High sparsity heads (most diffuse)
        self.local_window = 32  # Increased based on entropy analysis
        self.stride = 8
        self.num_global_tokens = 4
        self.max_seq_length = max_seq_length
        
        # Head distribution based on cluster sizes
        self.num_local_heads = 5  # Increased for Clusters 0 & 3 (largest clusters)
        self.num_strided_heads = 2  # For Cluster 1 (medium size)
        self.num_global_heads = 1  # For Cluster 2 (smallest cluster)
        
        # Entropy-based sparsity levels from cluster analysis
        self.sparsity_levels = {
            'local_low': 0.041,   # Cluster 0 (entropy: 1.821)
            'local_high': 0.136,  # Cluster 3 (entropy: 2.894)
            'strided': 0.316,     # Cluster 1 (entropy: 3.800)
            'global': 0.562       # Cluster 2 (entropy: 4.598)
        }
        
        # Target entropy values from cluster analysis
        self.target_entropy = {
            'local_low': 1.821,   # Cluster 0
            'local_high': 2.894,  # Cluster 3
            'strided': 3.800,     # Cluster 1
            'global': 4.598       # Cluster 2
        }
        
        # Cache for efficient computation
        self._mask_cache = {}
    
    def _compute_entropy(self, mask: torch.Tensor) -> torch.Tensor:
        """Compute attention entropy for the mask."""
        # Convert mask to probabilities
        probs = mask.float() / mask.sum(dim=-1, keepdim=True)
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        entropy = -(probs * (probs + eps).log()).sum(dim=-1).mean()
        return entropy
    
    def _adjust_mask_for_entropy(self, mask: torch.Tensor, target_entropy: float, 
                               max_iterations: int = 5) -> torch.Tensor:
        """Adjust mask to match target entropy."""
        current_entropy = self._compute_entropy(mask)
        
        # Binary search for sparsity adjustment
        left, right = 0.0, 1.0
        best_mask = mask.clone()
        
        for _ in range(max_iterations):
            if abs(current_entropy - target_entropy) < 0.1:
                break
                
            if current_entropy < target_entropy:
                # Need more connections
                sparsity = (left + right) / 2
                temp_mask = mask.clone()
                num_add = int(mask.size(-1) * sparsity)
                for i in range(mask.size(0)):
                    zero_indices = (~temp_mask[i]).nonzero().squeeze()
                    if len(zero_indices) > 0:
                        to_add = zero_indices[torch.randperm(len(zero_indices))[:num_add]]
                        temp_mask[i, to_add] = True
            else:
                # Need fewer connections
                sparsity = (left + right) / 2
                temp_mask = mask.clone()
                num_remove = int(mask.size(-1) * sparsity)
                for i in range(mask.size(0)):
                    one_indices = temp_mask[i].nonzero().squeeze()
                    if len(one_indices) > num_remove:
                        to_remove = one_indices[torch.randperm(len(one_indices))[num_remove:]]
                        temp_mask[i, to_remove] = False
            
            new_entropy = self._compute_entropy(temp_mask)
            if abs(new_entropy - target_entropy) < abs(current_entropy - target_entropy):
                best_mask = temp_mask.clone()
                current_entropy = new_entropy
            
            if new_entropy < target_entropy:
                left = sparsity
            else:
                right = sparsity
        
        return best_mask
    
    def _create_local_mask(self, seq_length: int) -> torch.Tensor:
        """Create local attention mask with sliding window."""
        mask = torch.zeros(seq_length, seq_length, dtype=torch.bool)
        
        # Create initial local window
        for i in range(seq_length):
            start = max(0, i - self.local_window)
            end = min(seq_length, i + self.local_window + 1)
            mask[i, start:end] = True
        
        # Adjust for target entropy - use lower entropy for first half of local heads
        head_idx = self.layer_idx % self.num_local_heads
        target_entropy = self.target_entropy['local_low'] if head_idx < self.num_local_heads // 2 \
                        else self.target_entropy['local_high']
        
        mask = self._adjust_mask_for_entropy(mask, target_entropy)
        return mask
    
    def _create_strided_mask(self, seq_length: int) -> torch.Tensor:
        """Create strided attention mask with entropy-based sparsity."""
        mask = torch.zeros(seq_length, seq_length, dtype=torch.bool)
        
        # Initial strided pattern
        stride_step = max(2, int(1 / (1 - self.sparsity_levels['strided'])))
        for i in range(seq_length):
            # Local window
            start = max(0, i - self.local_window // 2)
            end = min(seq_length, i + self.local_window // 2 + 1)
            mask[i, start:end] = True
            # Strided connections
            indices = torch.arange(i % stride_step, seq_length, stride_step)
            mask[i, indices] = True
        
        # Adjust for target entropy
        mask = self._adjust_mask_for_entropy(mask, self.target_entropy['strided'])
        return mask
    
    def _create_global_mask(self, seq_length: int) -> torch.Tensor:
        """Create global attention mask with high sparsity."""
        mask = torch.zeros(seq_length, seq_length, dtype=torch.bool)
        
        # Initial global pattern
        num_global = max(2, int(seq_length * (1 - self.sparsity_levels['global'])))
        step = max(1, seq_length // num_global)
        
        # Add global token connections
        global_indices = [0, seq_length - 1]  # Always include start and end
        if num_global > 2:
            global_indices.extend(list(range(step, seq_length - 1, step)))
        global_indices = sorted(list(set(global_indices)))[:num_global]
        
        # Allow attention to and from global tokens
        mask[:, global_indices] = True
        mask[global_indices, :] = True
        
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
        """Forward pass with entropy-based sparse attention."""
        batch_size, seq_length, _ = query.shape
        scaling = float(self.head_dim) ** -0.5
        
        # Linear projections and reshape
        q = self.q_proj(query).view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, seq_length, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, seq_length, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch_size, num_heads, seq_length, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Initialize output tensor
        output = torch.zeros_like(q)
        attn_weights_list = []
        
        # Process each head separately to save memory
        for head_idx in range(self.num_heads):
            # Get head-specific tensors
            q_head = q[:, head_idx:head_idx+1]  # Keep dim for broadcasting
            k_head = k[:, head_idx:head_idx+1]
            v_head = v[:, head_idx:head_idx+1]
            
            # Compute attention scores for this head
            attn_weights_head = torch.matmul(q_head * scaling, k_head.transpose(-2, -1))
            
            # Apply sparse attention pattern for this head
            head_mask = self._get_mask_for_head(head_idx, seq_length).to(query.device)
            attn_weights_head = attn_weights_head.masked_fill(~head_mask, float('-inf'))
            
            # Apply key padding mask if provided
            if key_padding_mask is not None:
                attn_weights_head = attn_weights_head.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    float('-inf')
                )
            
            # Compute softmax with improved numerical stability
            attn_weights_head = attn_weights_head - attn_weights_head.max(dim=-1, keepdim=True)[0].detach()
            attn_weights_head = F.softmax(attn_weights_head, dim=-1)
            
            # Apply dropout
            attn_weights_head = F.dropout(attn_weights_head, p=self.dropout, training=self.training)
            
            # Compute output for this head
            output[:, head_idx:head_idx+1] = torch.matmul(attn_weights_head, v_head)
            
            # Store attention weights if needed
            if attn_mask is not None:
                attn_weights_list.append(attn_weights_head)
        
        # Combine outputs from all heads
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_dim)
        output = self.out_proj(output)
        
        # Combine attention weights if needed
        if attn_mask is not None:
            attn_weights = torch.cat(attn_weights_list, dim=1)
        else:
            attn_weights = None
        
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
        
        self.d_model = d_model
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            SparseTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                layer_idx=i
            )
            for i in range(num_layers)
        ])
        
        # Output layer
        self.norm = nn.LayerNorm(d_model, eps=1e-4)
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters."""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.fc_out.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.fc_out.bias)
    
    def forward(self, src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            src: Source tensor [batch_size, seq_length]
            src_mask: Optional mask [seq_length, seq_length]
            src_key_padding_mask: Optional mask [batch_size, seq_length]
            
        Returns:
            Output tensor of shape [batch_size, seq_length, vocab_size]
        """
        # Embed tokens and positions
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        # Pass through layers
        for layer in self.layers:
            src = layer(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        
        # Output projection
        output = self.norm(src)
        output = self.fc_out(output)
        
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