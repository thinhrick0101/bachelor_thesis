import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sparse_attention import SparseAttention, AdaptiveSparseAttention
from torch.utils.checkpoint import checkpoint

class SparseTransformerEncoderLayer(nn.Module):
    """Sparse Transformer Encoder Layer that uses our sparse attention mechanism"""
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", layer_idx=0, use_adaptive=False):
        super().__init__()
        
        # Choose between regular sparse attention or adaptive sparse attention
        if use_adaptive:
            self.self_attn = AdaptiveSparseAttention(
                embed_dim=d_model,
                num_heads=nhead,
                layer_idx=layer_idx,
                dropout=dropout
            )
        else:
            self.self_attn = SparseAttention(
                embed_dim=d_model,
                num_heads=nhead,
                layer_idx=layer_idx,
                dropout=dropout
            )
        
        # Feed-forward network with gating
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear1_gate = nn.Linear(d_model, dim_feedforward)  # Gating layer
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer norms with stable epsilon
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        
        # Dropouts with different rates
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout * 1.5)  # Higher dropout for FFN

        # Layer scale parameters with careful initialization
        scale_init = 0.1 * (0.9 ** layer_idx)  # Decrease scale for deeper layers
        self.layer_scale1 = nn.Parameter(torch.ones(1, 1, d_model) * scale_init)
        self.layer_scale2 = nn.Parameter(torch.ones(1, 1, d_model) * scale_init)
        
        # Gradient scaling factors
        self.attn_scale = 1.0 / math.sqrt(d_model)
        self.ffn_scale = 1.0 / math.sqrt(dim_feedforward)

        self.activation = F.gelu  # Use GELU activation for better stability

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Pre-norm architecture
        src_norm = self.norm1(src)
        
        # Scale input to attention for stability
        src_norm = src_norm * self.attn_scale
        
        # Multi-head sparse attention
        src2, attn_weights = self.self_attn(
            query=src_norm,
            key=src_norm,
            value=src_norm,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        
        # Rescale attention output
        src2 = src2 / self.attn_scale
        
        # Apply layer scaling and residual with gradient scaling
        src = src + self.dropout1(self.layer_scale1 * src2)

        # Pre-norm for feed-forward
        src_norm = self.norm2(src)
        
        # Scale input to FFN for stability
        src_norm = src_norm * self.ffn_scale
        
        # Feed-forward network with gating mechanism
        gate = torch.sigmoid(self.linear1_gate(src_norm))
        src2 = self.linear1(src_norm)
        src2 = self.activation(src2) * gate
        src2 = self.linear2(self.dropout(src2))
        
        # Rescale FFN output
        src2 = src2 / self.ffn_scale
        
        # Apply layer scaling and residual
        src = src + self.dropout2(self.layer_scale2 * src2)

        return src, attn_weights

class SparseCharTransformer(nn.Module):
    """Character-level Transformer with sparse attention mechanisms"""
    
    def __init__(self, vocab_size=256, d_model=512, nhead=8, num_layers=12,
                 dim_feedforward=2048, dropout=0.1, activation="gelu",
                 use_adaptive_attention=False):
        super().__init__()
        
        self.d_model = d_model
        
        # Initialize embedding with careful scaling
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)
        
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Create encoder layers with progressive dropout and scaling
        self.layers = nn.ModuleList([
            SparseTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout * (1 + 0.05 * i),  # Gentler dropout progression
                activation=activation,
                layer_idx=i,
                use_adaptive=use_adaptive_attention
            ) for i in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(d_model, eps=1e-5)
        
        # Initialize output projection carefully
        self.fc_out = nn.Linear(d_model, vocab_size)
        nn.init.normal_(self.fc_out.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.fc_out.bias)
        
        # Gradient checkpointing state
        self.gradient_checkpointing = False
        
    def _reset_parameters(self):
        """Initialize parameters with better scaling"""
        # Initialize attention weights with smaller values
        for layer in self.layers:
            if hasattr(layer.self_attn, 'q_proj'):
                nn.init.normal_(layer.self_attn.q_proj.weight, mean=0.0, std=0.02)
                nn.init.normal_(layer.self_attn.k_proj.weight, mean=0.0, std=0.02)
                nn.init.normal_(layer.self_attn.v_proj.weight, mean=0.0, std=0.02)
                nn.init.zeros_(layer.self_attn.q_proj.bias)
                nn.init.zeros_(layer.self_attn.k_proj.bias)
                nn.init.zeros_(layer.self_attn.v_proj.bias)
            
            # Initialize FFN with smaller values
            nn.init.normal_(layer.linear1.weight, mean=0.0, std=0.02)
            nn.init.normal_(layer.linear2.weight, mean=0.0, std=0.02)
            nn.init.zeros_(layer.linear1.bias)
            nn.init.zeros_(layer.linear2.bias)
    
    def gradient_checkpointing_enable(self):
        """Enables gradient checkpointing for memory efficiency"""
        self.gradient_checkpointing = True
    
    def gradient_checkpointing_disable(self):
        """Disables gradient checkpointing"""
        self.gradient_checkpointing = False
    
    def _layer_forward(self, layer, src, src_mask=None, src_key_padding_mask=None):
        """Helper function for gradient checkpointing"""
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        if self.gradient_checkpointing:
            return checkpoint(create_custom_forward(layer), src, src_mask, src_key_padding_mask)
        return layer(src, src_mask, src_key_padding_mask)
                
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Scale embeddings carefully
        src = self.embedding(src) * 0.1  # Fixed small scale
        src = self.pos_encoder(src)
        
        # Store attention weights for analysis
        attention_weights = []
        
        # Pass through encoder layers
        for layer in self.layers:
            if self.gradient_checkpointing:
                src, attn_weights = self._layer_forward(layer, src, src_mask, src_key_padding_mask)
            else:
                src, attn_weights = layer(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
            attention_weights.append(attn_weights)
        
        # Final normalization
        output = self.norm(src)
        
        # Project to vocabulary size
        output = self.fc_out(output)
        
        return output, attention_weights
    
    def generate(self, prompt, max_length=1000, temperature=0.7, top_k=50, top_p=0.9,
                tokenizer=None, device='cuda'):
        """Generate text using the model"""
        self.eval()
        
        # Initialize sequence with prompt
        seq = prompt.clone()
        
        # Generate tokens
        for _ in range(max_length):
            # Create attention mask for generation
            src_mask = generate_square_subsequent_mask(seq.size(1)).to(device)
            
            # Get model predictions
            with torch.no_grad():
                output, _ = self(seq, src_mask=src_mask)
                
            # Get next token probabilities
            next_token_logits = output[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            seq = torch.cat([seq, next_token], dim=1)
            
            # Check for end of text token if tokenizer is provided
            if tokenizer and tokenizer.decode([next_token.item()]) == '<|endoftext|>':
                break
        
        return seq

class PositionalEncoding(nn.Module):
    """Positional encoding for the transformer"""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

def generate_square_subsequent_mask(sz):
    """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
    
    Args:
        sz: The sequence length
        
    Returns:
        A tensor of shape [sz, sz] where subsequent positions are masked (upper triangle)
    """
    mask = torch.triu(torch.ones(sz, sz), diagonal=1)  # Upper triangle above diagonal
    mask = mask.bool()  # Convert to boolean
    return mask 