import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional, Tuple
import math
import time

class AttentionPatternAnalyzer:
    """
    Analyzes attention patterns in transformer models to inform sparse attention design
    """
    def __init__(self, model):
        self.model = model
        self.attention_maps = []
        self.register_hooks()
        
    def register_hooks(self):
        """Register forward hooks to capture attention patterns"""
        def attention_hook(module, input, output):
            # Extract attention weights from output
            # Shape: [batch_size, num_heads, seq_len, seq_len]
            attn_weights = output[1] if isinstance(output, tuple) else output
            self.attention_maps.append(attn_weights.detach().cpu())
            
        # Register hooks for each attention layer
        for name, module in self.model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                module.register_forward_hook(attention_hook)
                
    def analyze_sequence(self, input_seq, tokenizer):
        """
        Analyze attention patterns for a given input sequence
        
        Args:
            input_seq: Input sequence to analyze
            tokenizer: Tokenizer for decoding positions
            
        Returns:
            Dictionary containing analysis results
        """
        self.attention_maps = []
        self.model.eval()
        
        with torch.no_grad():
            # Forward pass to collect attention maps
            _ = self.model(input_seq)
        
        results = {
            'global_patterns': self.analyze_global_patterns(),
            'local_patterns': self.analyze_local_patterns(),
            'head_specialization': self.analyze_head_specialization(),
            'sparsity_potential': self.calculate_sparsity_potential()
        }
        
        return results
    
    def analyze_global_patterns(self):
        """Analyze global attention patterns across all layers"""
        global_patterns = {}
        
        for layer_idx, attn_map in enumerate(self.attention_maps):
            # Average across batch and heads
            avg_attention = attn_map.mean(dim=(0, 1))
            
            # Calculate various metrics
            global_patterns[f'layer_{layer_idx}'] = {
                'mean_attention': avg_attention.mean().item(),
                'attention_entropy': self._calculate_entropy(avg_attention),
                'attention_sparsity': self._calculate_sparsity(avg_attention),
                'diagonal_bias': self._calculate_diagonal_bias(avg_attention)
            }
            
        return global_patterns
    
    def analyze_local_patterns(self):
        """Analyze local attention patterns and window sizes"""
        local_patterns = {}
        
        for layer_idx, attn_map in enumerate(self.attention_maps):
            avg_attention = attn_map.mean(dim=(0, 1))
            
            # Analyze effective window sizes
            window_sizes = self._find_effective_windows(avg_attention)
            local_patterns[f'layer_{layer_idx}'] = {
                'median_window': window_sizes['median'],
                'percentile_90_window': window_sizes['90th_percentile'],
                'local_concentration': self._calculate_local_concentration(avg_attention)
            }
            
        return local_patterns
    
    def analyze_head_specialization(self):
        """Analyze how different attention heads specialize"""
        head_specialization = {}
        
        for layer_idx, attn_map in enumerate(self.attention_maps):
            num_heads = attn_map.size(1)
            head_metrics = []
            
            for head_idx in range(num_heads):
                head_attention = attn_map[:, head_idx]
                head_metrics.append({
                    'sparsity': self._calculate_sparsity(head_attention.mean(0)),
                    'pattern_type': self._classify_attention_pattern(head_attention)
                })
            
            head_specialization[f'layer_{layer_idx}'] = head_metrics
            
        return head_specialization
    
    def calculate_sparsity_potential(self):
        """Calculate potential for sparsification based on attention patterns"""
        sparsity_metrics = {}
        
        for layer_idx, attn_map in enumerate(self.attention_maps):
            # Calculate attention weight distribution
            flattened_weights = attn_map.reshape(-1).numpy()
            
            # Calculate various sparsity metrics
            sparsity_metrics[f'layer_{layer_idx}'] = {
                'top_k_90': self._find_minimal_attention_keys(flattened_weights, 0.9),
                'effective_rank': self._calculate_effective_rank(attn_map),
                'compressibility': self._estimate_compressibility(attn_map)
            }
            
        return sparsity_metrics
    
    @staticmethod
    def _calculate_entropy(attention_weights):
        """Calculate entropy of attention distribution"""
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        normalized_weights = attention_weights + eps
        normalized_weights = normalized_weights / normalized_weights.sum()
        return -(normalized_weights * torch.log(normalized_weights)).sum().item()
    
    @staticmethod
    def _calculate_sparsity(attention_weights):
        """Calculate sparsity of attention weights"""
        threshold = 0.01  # Consider weights below this threshold as effectively zero
        return (attention_weights < threshold).float().mean().item()
    
    @staticmethod
    def _calculate_diagonal_bias(attention_weights):
        """Calculate bias towards diagonal attention"""
        seq_len = attention_weights.size(0)
        diagonal_mask = torch.eye(seq_len)
        return (attention_weights * diagonal_mask).sum().item() / attention_weights.sum().item()
    
    @staticmethod
    def _find_effective_windows(attention_weights):
        """Find effective attention window sizes"""
        seq_len = attention_weights.size(0)
        cumsum_weights = []
        
        for i in range(seq_len):
            # Calculate cumulative attention around each position
            center = attention_weights[i]
            distances = torch.arange(seq_len).float()
            sorted_weights, _ = torch.sort(center, descending=True)
            cumsum = torch.cumsum(sorted_weights, 0)
            
            # Find window sizes for different thresholds
            # Handle case where cumsum doesn't reach threshold
            threshold_indices = torch.where(cumsum >= 0.9)[0]
            if len(threshold_indices) > 0:
                window_90 = threshold_indices[0].item() + 1
            else:
                # If threshold is never reached, use full sequence length
                window_90 = seq_len
            
            cumsum_weights.append(window_90)
            
        if not cumsum_weights:  # If no valid windows found
            return {
                'median': seq_len,
                '90th_percentile': seq_len
            }
            
        return {
            'median': float(np.median(cumsum_weights)),
            '90th_percentile': float(np.percentile(cumsum_weights, 90))
        }
    
    @staticmethod
    def _calculate_local_concentration(attention_weights):
        """Calculate how much attention is concentrated locally"""
        seq_len = attention_weights.size(0)
        local_window = min(50, seq_len // 4)  # Define local window size
        
        local_attention = 0
        for i in range(seq_len):
            start_idx = max(0, i - local_window)
            end_idx = min(seq_len, i + local_window + 1)
            # Handle 1D tensor correctly
            local_attention += attention_weights[start_idx:end_idx].sum().item()
            
        return local_attention / attention_weights.sum().item()
    
    @staticmethod
    def _classify_attention_pattern(attention_weights):
        """Classify attention pattern type"""
        # Average across batch dimension
        avg_attention = attention_weights.mean(0)
        
        # Calculate various pattern metrics
        diagonal_strength = AttentionPatternAnalyzer._calculate_diagonal_bias(avg_attention)
        local_strength = AttentionPatternAnalyzer._calculate_local_concentration(avg_attention)
        sparsity = AttentionPatternAnalyzer._calculate_sparsity(avg_attention)
        
        # Classify based on metrics
        if diagonal_strength > 0.5:
            return 'diagonal'
        elif local_strength > 0.7:
            return 'local'
        elif sparsity > 0.8:
            return 'sparse'
        else:
            return 'global'
    
    @staticmethod
    def _find_minimal_attention_keys(weights, coverage_threshold):
        """Find minimum number of attention keys needed for desired coverage"""
        sorted_weights = np.sort(weights)[::-1]
        cumsum = np.cumsum(sorted_weights)
        total_sum = cumsum[-1]
        
        # Find index where cumulative sum reaches threshold
        idx = np.searchsorted(cumsum, coverage_threshold * total_sum)
        return idx + 1
    
    @staticmethod
    def _calculate_effective_rank(attention_weights):
        """Calculate effective rank of attention matrix"""
        # Ensure matrix is 2D
        if attention_weights.dim() == 4:  # [batch, heads, seq, seq]
            matrix = attention_weights.mean(dim=(0, 1))  # Average over batch and heads
        elif attention_weights.dim() == 1:  # [seq]
            seq_len = attention_weights.size(0)
            matrix = attention_weights.view(seq_len, 1)  # Reshape to [seq, 1]
        else:
            matrix = attention_weights
            
        # Add small epsilon to ensure numerical stability
        matrix = matrix + 1e-8
        
        # Calculate SVD
        try:
            U, S, V = torch.svd(matrix)
            
            # Calculate normalized singular values
            normalized_singular_values = S / S.sum()
            
            # Calculate effective rank
            return torch.exp(-torch.sum(normalized_singular_values * torch.log(normalized_singular_values + 1e-10))).item()
        except RuntimeError:
            # Fallback if SVD fails
            return 1.0  # Return minimum rank as fallback
    
    @staticmethod
    def _estimate_compressibility(attention_weights):
        """Estimate compressibility of attention patterns"""
        # Ensure matrix is 2D
        if attention_weights.dim() == 4:  # [batch, heads, seq, seq]
            matrix = attention_weights.mean(dim=(0, 1))  # Average over batch and heads
        elif attention_weights.dim() == 1:  # [seq]
            seq_len = attention_weights.size(0)
            matrix = attention_weights.view(seq_len, 1)  # Reshape to [seq, 1]
        else:
            matrix = attention_weights
            
        # Add small epsilon to ensure numerical stability
        matrix = matrix + 1e-8
        
        try:
            # Use ratio of top-k singular values as proxy for compressibility
            U, S, V = torch.svd(matrix)
            
            top_k = min(10, len(S))
            return (S[:top_k].sum() / S.sum()).item()
        except RuntimeError:
            # Fallback if SVD fails
            return 0.0  # Return minimum compressibility as fallback

class CustomSparseAttention(nn.Module):
    """
    Optimized sparse attention mechanism with adaptive patterns and efficient computation
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, 
                 local_window_size=50, num_global_tokens=10, sparsity_threshold=0.01):
        super(CustomSparseAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        # Initialize projections with careful scaling
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Better initialization for stable training
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(proj.weight, gain=1/math.sqrt(2))
            if bias:
                nn.init.zeros_(proj.bias)
        
        # Adaptive sparse attention parameters
        self.local_window_size = local_window_size
        self.num_global_tokens = num_global_tokens
        self.sparsity_threshold = sparsity_threshold
        
        # Learnable temperature with better initialization
        self.temperature = nn.Parameter(torch.ones(1) * math.sqrt(self.head_dim))
        
        # Cache for efficient computation
        self._mask_cache = {}
        
    def _create_sparse_mask(self, tgt_len: int, src_len: int, device: torch.device) -> torch.Tensor:
        """
        Create optimized sparse attention mask with efficient caching
        """
        cache_key = f"{tgt_len}_{src_len}"
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key].to(device)
            
        # Initialize mask efficiently
        mask = torch.zeros(tgt_len, src_len, dtype=torch.bool, device=device)
        
        # Compute adaptive window size based on sequence length
        adaptive_window = min(self.local_window_size, max(16, min(tgt_len, src_len) // 8))
        window_overlap = max(8, adaptive_window // 4)
        
        # Create local windows efficiently using broadcasting
        positions = torch.arange(tgt_len, device=device).unsqueeze(1)
        attend_to = torch.arange(src_len, device=device).unsqueeze(0)
        local_mask = (positions - attend_to).abs() <= adaptive_window // 2
        mask |= local_mask
        
        # Add strided windows efficiently
        if window_overlap > 0:
            stride_positions = torch.arange(0, tgt_len, adaptive_window - window_overlap, device=device)
            for pos in stride_positions:
                end_pos = min(pos + adaptive_window, tgt_len)
                if end_pos - pos > 1:
                    mask[pos:end_pos, max(0, pos-window_overlap):min(src_len, end_pos+window_overlap)] = True
        
        # Add global tokens using efficient tensor operations
        if self.num_global_tokens > 0 and src_len > self.num_global_tokens + 2:
            num_tokens = min(self.num_global_tokens, src_len // 4)
            if num_tokens > 0:
                # Use strided attention for global tokens
                stride = max(1, src_len // (num_tokens + 1))
                global_idx = torch.arange(stride, src_len-stride, stride, device=device)
                global_idx = global_idx[:num_tokens]
                if len(global_idx) > 0:
                    mask[:, global_idx] = True
        
        # Add diagonal attention efficiently
        diag_mask = torch.eye(tgt_len, src_len, dtype=torch.bool, device=device)
        mask |= diag_mask
        
        # Add neighbor attention efficiently
        if tgt_len > 1 and src_len > 1:
            neighbor_mask = torch.zeros_like(mask)
            neighbor_mask[:-1, 1:] = torch.eye(tgt_len-1, src_len-1, dtype=torch.bool, device=device)
            neighbor_mask[1:, :-1] = torch.eye(tgt_len-1, src_len-1, dtype=torch.bool, device=device)
            mask |= neighbor_mask
        
        # Cache the mask on CPU to save GPU memory
        self._mask_cache[cache_key] = mask.cpu()
        
        return mask
        
    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        batch_size, tgt_len, embed_dim = query.size()
        src_len = key.size(1)
        
        scaling = float(self.head_dim) ** -0.5
        
        # Fused QKV projection for efficiency
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Efficient reshape and transpose
        q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2) * scaling
        k = k.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores with adaptive temperature
        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        attn_weights = attn_weights * torch.sigmoid(self.temperature)
        
        # Apply sparse attention pattern
        sparse_mask = self._create_sparse_mask(tgt_len, src_len, query.device)
        attn_weights = attn_weights.masked_fill(~sparse_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Handle attention masks
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_weights = attn_weights.masked_fill(~attn_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            else:
                attn_weights = attn_weights + attn_mask.unsqueeze(0).unsqueeze(0)
                
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        # Efficient softmax with improved numerical stability
        attn_weights = attn_weights - attn_weights.max(dim=-1, keepdim=True)[0].detach()
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Adaptive dropout based on sparsity
        if self.training and self.dropout > 0:
            sparsity = (~sparse_mask).float().mean().item()
            effective_dropout = self.dropout * (1.0 - sparsity)
            attn_weights = F.dropout(attn_weights, p=effective_dropout, training=True)
        
        # Compute output efficiently
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).reshape(batch_size, tgt_len, embed_dim)
        output = self.out_proj(output)
        
        if need_weights:
            return output, attn_weights
        else:
            return output, None

def visualize_attention_patterns(analyzer_results, save_path='attention_analysis.png'):
    """
    Visualize attention pattern analysis results
    """
    num_layers = len(analyzer_results['global_patterns'])
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # Plot 1: Global attention patterns across layers
    ax = axes[0, 0]
    layers = list(range(num_layers))
    mean_attention = [analyzer_results['global_patterns'][f'layer_{i}']['mean_attention'] for i in layers]
    entropy = [analyzer_results['global_patterns'][f'layer_{i}']['attention_entropy'] for i in layers]
    
    ax.plot(layers, mean_attention, 'b-', label='Mean Attention')
    ax.plot(layers, entropy, 'r--', label='Entropy')
    ax.set_title('Global Attention Patterns')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Value')
    ax.legend()
    
    # Plot 2: Local attention windows
    ax = axes[0, 1]
    median_windows = [analyzer_results['local_patterns'][f'layer_{i}']['median_window'] for i in layers]
    percentile_90 = [analyzer_results['local_patterns'][f'layer_{i}']['percentile_90_window'] for i in layers]
    
    ax.plot(layers, median_windows, 'g-', label='Median Window')
    ax.plot(layers, percentile_90, 'm--', label='90th Percentile')
    ax.set_title('Local Attention Windows')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Window Size')
    ax.legend()
    
    # Plot 3: Head specialization
    ax = axes[1, 0]
    pattern_types = ['diagonal', 'local', 'sparse', 'global']
    pattern_counts = {pattern: [] for pattern in pattern_types}
    
    for layer in range(num_layers):
        layer_patterns = analyzer_results['head_specialization'][f'layer_{layer}']
        counts = {pattern: sum(1 for head in layer_patterns if head['pattern_type'] == pattern)
                 for pattern in pattern_types}
        for pattern in pattern_types:
            pattern_counts[pattern].append(counts[pattern])
    
    bottom = np.zeros(num_layers)
    for pattern in pattern_types:
        ax.bar(layers, pattern_counts[pattern], bottom=bottom, label=pattern)
        bottom += pattern_counts[pattern]
    
    ax.set_title('Head Specialization')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Number of Heads')
    ax.legend()
    
    # Plot 4: Sparsity potential
    ax = axes[1, 1]
    top_k_90 = [analyzer_results['sparsity_potential'][f'layer_{i}']['top_k_90'] for i in layers]
    effective_rank = [analyzer_results['sparsity_potential'][f'layer_{i}']['effective_rank'] for i in layers]
    
    ax1 = ax
    ax2 = ax1.twinx()
    
    l1 = ax1.plot(layers, top_k_90, 'c-', label='Top-k (90% coverage)')
    l2 = ax2.plot(layers, effective_rank, 'y--', label='Effective Rank')
    
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Top-k Keys')
    ax2.set_ylabel('Effective Rank')
    
    # Combine legends
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels)
    ax.set_title('Sparsity Potential')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def compare_attention_mechanisms(model, sparse_model, test_data, tokenizer):
    """
    Compare performance between full and sparse attention
    
    Args:
        model: Original model with full attention
        sparse_model: Model with sparse attention
        test_data: Test dataset
        tokenizer: Tokenizer for processing text
        
    Returns:
        Dictionary containing comparison metrics
    """
    results = {
        'perplexity': {'full': [], 'sparse': []},
        'speed': {'full': [], 'sparse': []},
        'memory': {'full': [], 'sparse': []}
    }
    
    models = {'full': model, 'sparse': sparse_model}
    device = next(model.parameters()).device  # Get device from model
    
    for name, m in models.items():
        m.eval()
        total_tokens = 0
        total_loss = 0
        
        with torch.no_grad():
            start_time = time.time()
            peak_memory = 0
            
            for batch in test_data:
                inputs, targets = batch
                # Move tensors to the same device as model
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Record memory usage
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                
                # Forward pass
                outputs = m(inputs)
                loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                
                # Update metrics
                total_tokens += targets.numel()
                total_loss += loss.item() * targets.numel()
                
                # Record peak memory
                if torch.cuda.is_available():
                    peak_memory = max(peak_memory, torch.cuda.max_memory_allocated())
            
            # Calculate metrics
            perplexity = math.exp(total_loss / total_tokens)
            speed = total_tokens / (time.time() - start_time)
            memory = peak_memory / 1024 / 1024 if torch.cuda.is_available() else 0
            
            # Store results
            results['perplexity'][name].append(perplexity)
            results['speed'][name].append(speed)
            results['memory'][name].append(memory)
    
    return results

def plot_comparison_results(results, save_path='attention_comparison.png'):
    """
    Plot comparison results between full and sparse attention
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot perplexity comparison
    ax = axes[0]
    perplexities = [results['perplexity']['full'][0], results['perplexity']['sparse'][0]]
    ax.bar(['Full Attention', 'Sparse Attention'], perplexities)
    ax.set_title('Perplexity Comparison')
    ax.set_ylabel('Perplexity')
    
    # Plot speed comparison
    ax = axes[1]
    speeds = [results['speed']['full'][0], results['speed']['sparse'][0]]
    ax.bar(['Full Attention', 'Sparse Attention'], speeds)
    ax.set_title('Speed Comparison')
    ax.set_ylabel('Tokens/second')
    
    # Plot memory comparison
    ax = axes[2]
    memory = [results['memory']['full'][0], results['memory']['sparse'][0]]
    ax.bar(['Full Attention', 'Sparse Attention'], memory)
    ax.set_title('Memory Usage')
    ax.set_ylabel('Memory (MB)')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close() 