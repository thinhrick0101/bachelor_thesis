import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from torch import Tensor
import warnings

try:
    import xformers.ops as xops
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xformers not available. Using optimized block-sparse implementation.")

@torch.jit.script
def compute_block_attention_cpu(q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """
    JIT-optimized block attention computation for CPU tensors.
    (We only JIT-script the CPU path to avoid extra CUDA buffers.)
    """
    scores = torch.matmul(q, k.transpose(-2, -1))
    if mask is not None:
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).expand(q.size(0), -1, -1)
        scores = scores.masked_fill(~mask.to(torch.bool), -10000.0)
    attn_probs = F.softmax(scores, dim=-1, dtype=torch.float32)
    attn_probs = attn_probs.to(dtype=scores.dtype)
    if attn_probs.size(-1) != v.size(1):
        raise RuntimeError(f"Attention weights dim {attn_probs.size(-1)} does not match value dim {v.size(1)}")
    return torch.matmul(attn_probs, v)

def compute_block_attention_cuda(q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """
    Block attention on CUDA—NOT JIT-scripted, so PyTorch can manage memory more flexibly.
    """
    # scores might be large, but we hope to stay under memory by making block_size small
    scores = torch.matmul(q, k.transpose(-2, -1))
    if mask is not None:
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).expand(q.size(0), -1, -1)
        scores = scores.masked_fill(~mask.to(torch.bool), -10000.0)

    # Numerically stable softmax
    scores_max = torch.max(scores, dim=-1, keepdim=True)[0]
    scores = scores - scores_max
    attn_probs = torch.exp(scores)
    attn_probs = attn_probs / (torch.sum(attn_probs, dim=-1, keepdim=True) + 1e-6)

    if attn_probs.size(-1) != v.size(1):
        raise RuntimeError(f"Attn weights length {attn_probs.size(-1)} vs value length {v.size(1)}")

    return torch.matmul(attn_probs, v)

def create_pattern_mask(seq_len: int, window: int, stride: int, 
                        is_global: bool, device: torch.device) -> Tensor:
    """
    JIT-uncompiled pattern mask creation—PyTorch will handle it dynamically.
    """
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
    if window > 0:
        positions = torch.arange(seq_len, device=device)
        dist = positions.unsqueeze(1) - positions.unsqueeze(0)
        mask |= dist.abs() <= window // 2
    if stride > 0:
        stride_positions = torch.arange(0, seq_len, stride, device=device)
        mask[:, stride_positions] = True
    if is_global:
        global_positions = torch.tensor([0, seq_len // 2, seq_len - 1], device=device)
        mask[:, global_positions] = True
    return mask

class SparseMultiHeadAttention(nn.Module):
    """
    Optimized sparse multi-head attention with smaller block_size, no CUDA-graph capture,
    and JIT only on the CPU fallback.
    """
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embedding_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        assert self.head_dim * num_heads == embedding_dim, "embedding_dim must be divisible by num_heads"
        assert num_heads % 4 == 0, "num_heads must be divisible by 4 for cluster patterns"

        # Projections
        self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)

        # We only keep 4 patterns; you can comment out if you need fewer
        self.heads_per_pattern = num_heads // 4
        self.patterns = {
            'narrow_local':   {'window': 8,  'stride': 0},
            'wide_strided':   {'window': 32, 'stride': 8},
            'global_strided': {'window': 0,  'stride': 32},
            'wide_local':     {'window': 32, 'stride': 0},
        }

        # Reduce block_size from 32 → 16 (or even 8) to save peak memory
        self.block_size = 16

        # Cache for masks
        self._mask_cache = {}

    def _combine_masks(
        self,
        pattern_mask: Tensor,
        attn_mask: Optional[Tensor],
        padding_mask: Optional[Tensor],
        i: int,
        end_i: int,
        j: int,
        end_j: int,
    ) -> Tensor:
        """
        Combine pattern_mask, causal/padding masks blockwise, in bool format.
        """
        combined = pattern_mask[i:end_i, j:end_j].clone()
        if attn_mask is not None:
            block_attn = attn_mask[i:end_i, j:end_j].to(torch.bool)
            combined &= block_attn
        if padding_mask is not None:
            # padding_mask: [B, L] → [B, 1, 1, L] for broadcast
            pad = padding_mask.view(padding_mask.size(0), 1, 1, -1)
            pad_block = pad[..., j:end_j].to(torch.bool)  # [B, 1, 1, block_j]
            combined = combined.unsqueeze(0) & pad_block  # [B, block_i, block_j]
        return combined

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
        need_weights: bool = False,
        attn_padding_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        query/key/value: [B, L, E]
        attn_mask: optional [L, L]
        attn_padding_mask: optional [B, L]
        """
        B, L, _ = query.size()
        device = query.device
        is_cuda = query.is_cuda

        # 1) Project and reshape → [B, heads, L, head_dim]
        q = self.q_proj(query).view(B, L, self.num_heads, self.head_dim).transpose(1, 2) / self.scale
        k = self.k_proj(key).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        if XFORMERS_AVAILABLE and is_cuda:
            # Let xFormers do it if possible
            out = xops.memory_efficient_attention(
                q, k, v,
                attn_bias=xops.LowerTriangularMask() if attn_mask is None else attn_mask.to(torch.bool),
                p=self.dropout if self.training else 0.0
            )
            # [B, H, L, D] → transpose → [B, L, H, D] → [B, L, E]
            out = out.transpose(1, 2).reshape(B, L, self.embedding_dim)
            out = self.out_proj(out)
            return out, None

        # 2) Otherwise, do block‑sparse ourselves with smaller blocks
        attn_output = torch.zeros_like(q)

        for idx, (pattern_name, pat) in enumerate(self.patterns.items()):
            start_h = idx * self.heads_per_pattern
            end_h = (idx + 1) * self.heads_per_pattern

            cache_key = f"{pattern_name}_{L}"
            if cache_key not in self._mask_cache:
                self._mask_cache[cache_key] = create_pattern_mask(
                    L, pat["window"], pat["stride"], pattern_name == "global_strided", device
                )
            pat_mask = self._mask_cache[cache_key]  # [L, L], bool

            for i in range(0, L, self.block_size):
                i_end = min(i + self.block_size, L)
                for j in range(0, L, self.block_size):
                    j_end = min(j + self.block_size, L)

                    # If no overlap in the block, skip
                    if not pat_mask[i:i_end, j:j_end].any():
                        continue

                    # slice per-pattern heads
                    q_blk = q[:, start_h:end_h, i:i_end].contiguous()  # [B, heads_per_pat, b_i, head_dim]
                    k_blk = k[:, start_h:end_h, j:j_end].contiguous()
                    v_blk = v[:, start_h:end_h, j:j_end].contiguous()

                    # build the combined mask block
                    blk_mask = self._combine_masks(pat_mask, attn_mask, attn_padding_mask, i, i_end, j, j_end)
                    # blk_mask: either [b_i, b_j] (no batch) or [B, 1, 1, b_j] (with padding mask)

                    # reshape for compute
                    Hblk = end_h - start_h
                    q2 = q_blk.view(B * Hblk, i_end - i, self.head_dim)
                    k2 = k_blk.view(B * Hblk, j_end - j, self.head_dim)
                    v2 = v_blk.view(B * Hblk, j_end - j, self.head_dim)

                    # run either CUDA or CPU version
                    if is_cuda:
                        attn_blk = compute_block_attention_cuda(q2, k2, v2, blk_mask)
                    else:
                        attn_blk = compute_block_attention_cpu(q2, k2, v2, blk_mask)

                    # reshape back to [B, heads_per_pat, b_i, head_dim]
                    attn_blk = attn_blk.view(B, Hblk, i_end - i, self.head_dim)
                    attn_output[:, start_h:end_h, i:i_end] = attn_blk

        # optional dropout on attention_output
        if self.training and self.dropout > 0:
            attn_output = F.dropout(attn_output, p=self.dropout, training=True)

        # 3) project out
        out = attn_output.transpose(1, 2).reshape(B, L, self.embedding_dim)
        out = self.out_proj(out)
        return out, None
