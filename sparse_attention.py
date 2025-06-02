"""
Optimized sparse attention implementation with block-sparse patterns and efficient CUDA execution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import warnings
from torch import Tensor

try:
    import xformers.ops as xops
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xformers not available. Using optimized block-sparse implementation.")

@torch.jit.script
def compute_block_attention(q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """JIT-optimized block attention computation"""
    # Ensure inputs are 3D: [batch_size * num_heads, seq_len, head_dim]
    batch_dim = q.size(0)
    q_len = q.size(1)
    k_len = k.size(1)
    
    # Compute attention scores with optimized CUDA execution
    with torch.cuda.amp.autocast(enabled=q.is_cuda):
        scores = torch.matmul(q, k.transpose(-2, -1))  # [batch_size * num_heads, q_len, k_len]
        
        if mask is not None:
            # Expand mask to match scores dimensions
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).expand(batch_dim, -1, -1)
            scores = scores.masked_fill(~mask.to(torch.bool), -10000.0)
        
        # Compute attention probabilities with improved numerical stability
        scores_max = torch.max(scores, dim=-1, keepdim=True)[0]
        scores = scores - scores_max
        attn_probs = torch.exp(scores)
        attn_probs = attn_probs / (torch.sum(attn_probs, dim=-1, keepdim=True) + 1e-6)
        
        if attn_probs.size(-1) != v.size(1):
            raise RuntimeError(f"Attention weights dim {attn_probs.size(-1)} does not match value dim {v.size(1)}")
        
        # Apply attention to values
        return torch.matmul(attn_probs, v)  # [batch_size * num_heads, q_len, head_dim]

@torch.jit.script
def create_pattern_mask(seq_len: int, window: int, stride: int, 
                       is_global: bool, device: torch.device) -> Tensor:
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

class SparseMultiHeadAttention(nn.Module):
    """
    Optimized sparse multi-head attention with efficient CUDA execution
    """
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embedding_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.block_size = 32  # Optimize for GPU memory access

        assert self.head_dim * num_heads == embedding_dim, "embedding_dim must be divisible by num_heads"
        assert num_heads % 4 == 0, "num_heads must be divisible by 4 for cluster patterns"

        # Q/K/V projections for all heads at once
        self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        
        # Initialize attention patterns
        self.heads_per_pattern = num_heads // 4
        self.patterns = {
            'narrow_local': {'window': 8, 'stride': 0},     # ±8 local window
            'wide_strided': {'window': 32, 'stride': 8},    # ±16 + strided/8
            'global_strided': {'window': 0, 'stride': 32},  # Global + strided/32
            'wide_local': {'window': 32, 'stride': 0}       # ±16 local window
        }
        
        # Cache for pattern masks
        self._mask_cache = {}
        
        # Register CUDA graph if available
        self._cuda_graph = None if not torch.cuda.is_available() else {}

    def _combine_masks(self, pattern_mask: Tensor, 
                      attn_mask: Optional[Tensor] = None,
                      padding_mask: Optional[Tensor] = None,
                      i: int = 0, end_i: int = 0, 
                      j: int = 0, end_j: int = 0) -> Tensor:
        """Safely combine masks with optimized CUDA execution"""
        with torch.cuda.amp.autocast(enabled=pattern_mask.is_cuda):
            # Start with pattern mask (always bool)
            combined_mask = pattern_mask[i:end_i, j:end_j].clone()
            
            # Add attention mask if provided
            if attn_mask is not None:
                mask_block = attn_mask[i:end_i, j:end_j].to(torch.bool)
                combined_mask &= mask_block
                
            # Add padding mask if provided
            if padding_mask is not None:
                pad_mask = padding_mask.view(padding_mask.size(0), 1, 1, -1)
                pad_block = pad_mask[..., j:end_j].to(torch.bool)
                combined_mask = combined_mask.unsqueeze(0) & pad_block
            
            return combined_mask

    @torch.jit.ignore
    def _maybe_capture_cuda_graph(self, key: str, q: Tensor, k: Tensor, v: Tensor,
                                block_mask: Tensor) -> Optional[Tuple[torch.cuda.CUDAGraph, Tensor]]:
        """Capture CUDA graph for repeated computations if possible"""
        if not torch.cuda.is_available() or not q.is_cuda:
            return None
            
        if key not in self._cuda_graph:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                output = compute_block_attention(q, k, v, block_mask)
            self._cuda_graph[key] = (g, output)
            
        return self._cuda_graph[key]

    def forward(self, query: Tensor, key: Tensor, value: Tensor,
               attn_mask: Optional[Tensor] = None,
               need_weights: bool = False,
               attn_padding_mask: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Efficient block-sparse attention with optimized CUDA execution
        """
        bsz, seq_len, _ = query.size()
        device = query.device

        # Project and reshape Q/K/V with optimized memory layout
        q = self.q_proj(query).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2) / self.scale
        k = self.k_proj(key).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if XFORMERS_AVAILABLE:
            # Use xformers if available
            with torch.cuda.amp.autocast(enabled=query.is_cuda):
                attn_output = xops.memory_efficient_attention(
                    q, k, v,
                    attn_bias=xops.LowerTriangularMask() if attn_mask is None else attn_mask.to(torch.bool),
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
                if cache_key not in self._mask_cache:
                    mask = create_pattern_mask(
                        seq_len, pattern['window'], pattern['stride'],
                        pattern_name == 'global_strided', device
                    )
                    self._mask_cache[cache_key] = mask
                
                pattern_mask = self._mask_cache[cache_key]
                
                # Process blocks for this pattern
                for i in range(0, seq_len, self.block_size):
                    end_i = min(i + self.block_size, seq_len)
                    for j in range(0, seq_len, self.block_size):
                        end_j = min(j + self.block_size, seq_len)
                        
                        if pattern_mask[i:end_i, j:end_j].any():
                            # Reshape blocks for batch computation
                            q_block = q[:, start_head:end_head, i:end_i].contiguous()
                            k_block = k[:, start_head:end_head, j:end_j].contiguous()
                            v_block = v[:, start_head:end_head, j:end_j].contiguous()
                            
                            # Combine masks safely for mixed precision
                            block_mask = self._combine_masks(
                                pattern_mask, attn_mask, attn_padding_mask,
                                i, end_i, j, end_j
                            )
                            
                            # Reshape for efficient computation
                            head_count = end_head - start_head
                            q_block = q_block.view(bsz * head_count, end_i - i, self.head_dim)
                            k_block = k_block.view(bsz * head_count, end_j - j, self.head_dim)
                            v_block = v_block.view(bsz * head_count, end_j - j, self.head_dim)
                            
                            # Try to use CUDA graph for repeated computations
                            graph_key = f"{pattern_name}_{i}_{j}_{seq_len}"
                            graph_result = self._maybe_capture_cuda_graph(
                                graph_key, q_block, k_block, v_block, block_mask
                            )
                            
                            if graph_result is not None:
                                graph, cached_output = graph_result
                                graph.replay()
                                block_output = cached_output
                            else:
                                # Compute attention and reshape back
                                block_output = compute_block_attention(q_block, k_block, v_block, block_mask)
                            
                            block_output = block_output.view(bsz, head_count, end_i - i, self.head_dim)
                            attn_output[:, start_head:end_head, i:end_i] = block_output

            # Apply dropout with CUDA optimization
            if self.training and self.dropout > 0:
                attn_output = F.dropout(attn_output, p=self.dropout, training=True)

        # Final output projection
        output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.embedding_dim)
        output = self.out_proj(output)

        return output, None



