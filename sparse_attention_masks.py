import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, Dict

class StaticSparseMask:
    """Generate and apply static sparse attention masks based on cluster patterns."""
    
    def __init__(self, 
                 max_seq_length: int,
                 num_heads: int,
                 local_window: int = 16,
                 stride: int = 8,
                 num_global_tokens: int = 4,
                 device: str = 'cuda'):
        """
        Initialize static sparse attention mask generator.
        
        Args:
            max_seq_length: Maximum sequence length
            num_heads: Number of attention heads
            local_window: Size of local attention window (one-sided)
            stride: Stride for periodic attention
            num_global_tokens: Number of global tokens for global attention
            device: Device to create masks on
        """
        self.max_seq_length = max_seq_length
        self.num_heads = num_heads
        self.local_window = local_window
        self.stride = stride
        self.num_global_tokens = num_global_tokens
        self.device = device
        
        # Pre-compute static masks
        self.masks = self._create_static_masks()
    
    def _create_local_mask(self) -> torch.Tensor:
        """Create local attention mask with sliding window."""
        mask = torch.zeros(self.max_seq_length, self.max_seq_length, dtype=torch.bool)
        for i in range(self.max_seq_length):
            start = max(0, i - self.local_window)
            end = min(self.max_seq_length, i + self.local_window + 1)
            mask[i, start:end] = True
        return mask
    
    def _create_strided_mask(self) -> torch.Tensor:
        """Create strided attention mask with periodic connections."""
        mask = torch.zeros(self.max_seq_length, self.max_seq_length, dtype=torch.bool)
        for i in range(self.max_seq_length):
            # Add local window
            start_local = max(0, i - self.local_window // 2)
            end_local = min(self.max_seq_length, i + (self.local_window // 2) + 1)
            mask[i, start_local:end_local] = True
            
            # Add strided connections
            strided_indices = torch.arange(0, self.max_seq_length, self.stride)
            mask[i, strided_indices] = True
        return mask
    
    def _create_global_mask(self) -> torch.Tensor:
        """Create global attention mask with fixed anchor tokens."""
        mask = torch.zeros(self.max_seq_length, self.max_seq_length, dtype=torch.bool)
        
        # Select global tokens (e.g., start, end, and evenly spaced tokens)
        global_indices = [0, self.max_seq_length - 1]  # Always include start and end
        if self.num_global_tokens > 2:
            step = self.max_seq_length // (self.num_global_tokens - 1)
            global_indices.extend(list(range(step, self.max_seq_length - 1, step)))
        global_indices = sorted(list(set(global_indices)))[:self.num_global_tokens]
        
        # Allow attention to and from global tokens
        mask[:, global_indices] = True  # Every token can attend to global tokens
        mask[global_indices, :] = True  # Global tokens can attend to every token
        return mask
    
    def _create_random_mask(self, sparsity: float = 0.8) -> torch.Tensor:
        """Create random attention mask with controlled sparsity."""
        mask = torch.zeros(self.max_seq_length, self.max_seq_length, dtype=torch.bool)
        
        # Ensure each token attends to itself
        mask.fill_diagonal_(True)
        
        # Randomly select connections while maintaining symmetry
        num_connections = int((1 - sparsity) * self.max_seq_length * self.max_seq_length)
        upper_indices = torch.triu_indices(self.max_seq_length, self.max_seq_length, offset=1)
        perm = torch.randperm(upper_indices.size(1))
        selected = perm[:num_connections // 2]
        
        # Set symmetric connections
        mask[upper_indices[0, selected], upper_indices[1, selected]] = True
        mask[upper_indices[1, selected], upper_indices[0, selected]] = True
        return mask
    
    def _create_static_masks(self) -> Dict[str, torch.Tensor]:
        """Create all static attention masks."""
        masks = {
            'local': self._create_local_mask(),
            'strided': self._create_strided_mask(),
            'global': self._create_global_mask(),
            'random': self._create_random_mask()
        }
        return {name: mask.to(self.device) for name, mask in masks.items()}
    
    def get_mask_for_head(self, head_idx: int, seq_length: int) -> torch.Tensor:
        """Get appropriate mask for a specific attention head."""
        # Assign masks based on head index (example assignment)
        if head_idx < 4:  # First 4 heads use local attention
            mask = self.masks['local']
        elif head_idx < 6:  # Next 2 heads use strided attention
            mask = self.masks['strided']
        elif head_idx < 7:  # One head uses global attention
            mask = self.masks['global']
        else:  # Remaining heads use random attention
            mask = self.masks['random']
            
        # Trim mask to actual sequence length
        return mask[:seq_length, :seq_length]
    
    def apply_mask(self, 
                  attention_scores: torch.Tensor,
                  head_idx: int,
                  key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply sparse attention mask to attention scores.
        
        Args:
            attention_scores: Raw attention scores [batch_size, num_heads, seq_length, seq_length]
            head_idx: Index of the current attention head
            key_padding_mask: Optional padding mask [batch_size, seq_length]
            
        Returns:
            Masked attention scores
        """
        batch_size, _, seq_length, _ = attention_scores.shape
        
        # Get appropriate sparse mask for this head
        sparse_mask = self.get_mask_for_head(head_idx, seq_length)
        
        # Combine with padding mask if provided
        if key_padding_mask is not None:
            padding_mask = ~key_padding_mask.bool().unsqueeze(1).unsqueeze(2)
            sparse_mask = sparse_mask & padding_mask
        
        # Apply mask
        masked_scores = attention_scores.masked_fill(~sparse_mask, float('-inf'))
        return masked_scores

class SparseAttentionHead(nn.Module):
    """Attention head with static sparse attention patterns."""
    
    def __init__(self,
                 embed_dim: int,
                 head_dim: int,
                 dropout: float = 0.1,
                 max_seq_length: int = 1024,
                 head_idx: int = 0):
        super().__init__()
        
        self.head_idx = head_idx
        self.head_dim = head_dim
        self.scaling = float(head_dim) ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize sparse mask generator
        self.sparse_mask = StaticSparseMask(
            max_seq_length=max_seq_length,
            num_heads=1,  # Each head has its own mask
            device=next(self.parameters()).device
        )
    
    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute sparse attention.
        
        Args:
            query: Query tensor [batch_size, seq_length, embed_dim]
            key: Key tensor [batch_size, seq_length, embed_dim]
            value: Value tensor [batch_size, seq_length, embed_dim]
            key_padding_mask: Optional padding mask [batch_size, seq_length]
            
        Returns:
            output: Attention output [batch_size, seq_length, head_dim]
            attention_weights: Attention weights [batch_size, seq_length, seq_length]
        """
        batch_size, seq_length, _ = query.shape
        
        # Project inputs
        q = self.q_proj(query) * self.scaling  # [batch_size, seq_length, head_dim]
        k = self.k_proj(key)                   # [batch_size, seq_length, head_dim]
        v = self.v_proj(value)                 # [batch_size, seq_length, head_dim]
        
        # Compute attention scores
        attention_scores = torch.bmm(q, k.transpose(1, 2))  # [batch_size, seq_length, seq_length]
        
        # Apply sparse attention mask
        masked_scores = self.sparse_mask.apply_mask(
            attention_scores.unsqueeze(1),  # Add head dimension
            self.head_idx,
            key_padding_mask
        ).squeeze(1)  # Remove head dimension
        
        # Apply softmax and dropout
        attention_weights = F.softmax(masked_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Compute output
        output = torch.bmm(attention_weights, v)
        
        return output, attention_weights

def convert_to_sparse_transformer(model, max_seq_length: int = 1024):
    """Convert a regular Transformer to use sparse attention patterns."""
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            # Get parent module
            parent_name = '.'.join(name.split('.')[:-1])
            parent = model
            for part in parent_name.split('.'):
                if part:
                    parent = getattr(parent, part)
            
            # Create new sparse attention heads
            embed_dim = module.embed_dim
            num_heads = module.num_heads
            head_dim = embed_dim // num_heads
            dropout = module.dropout
            
            sparse_heads = nn.ModuleList([
                SparseAttentionHead(
                    embed_dim=embed_dim,
                    head_dim=head_dim,
                    dropout=dropout,
                    max_seq_length=max_seq_length,
                    head_idx=i
                )
                for i in range(num_heads)
            ])
            
            # Replace the attention module
            setattr(parent, name.split('.')[-1], sparse_heads)
    
    return model 