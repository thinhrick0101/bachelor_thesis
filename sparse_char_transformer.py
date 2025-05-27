import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sparse_attention import SparseAttention, AdaptiveSparseAttention
from torch.utils.checkpoint import checkpoint

class SparseTransformerEncoderLayer(nn.Module):
    """Sparse Transformer Encoder Layer with proper sparse attention"""
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="gelu", layer_idx=0, use_adaptive=False):
        super().__init__()
        
        # Use proper sparse attention
        if use_adaptive:
            self.self_attn = AdaptiveSparseAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout
            )
        else:
            self.self_attn = SparseAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout
            )
        
        # Feed-forward network with better initialization
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.gelu if activation == "gelu" else F.relu

        # Initialize weights with better scaling
        with torch.no_grad():
            nn.init.xavier_uniform_(self.linear1.weight, gain=0.2)
            nn.init.xavier_uniform_(self.linear2.weight, gain=0.2)
            nn.init.zeros_(self.linear1.bias)
            nn.init.zeros_(self.linear2.bias)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Pre-norm architecture
        src2 = self.norm1(src)
        
        # Self attention with sparse pattern
        src2, attn_weights = self.self_attn(
            src2, src2, src2,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        
        # Residual connection with dropout
        src = src + self.dropout1(src2)
        
        # Pre-norm for FFN
        src2 = self.norm2(src)
        
        # FFN with less aggressive clamping
        src2 = self.linear1(src2)
        src2 = self.activation(src2)
        src2 = torch.clamp(src2, min=-3.0, max=3.0)  # Less restrictive clamping
        src2 = self.dropout(src2)
        src2 = self.linear2(src2)
        
        # Residual connection with dropout
        src = src + self.dropout2(src2)

        return src, attn_weights

class SparseCharTransformer(nn.Module):
    """Character-level Transformer with Sparse Attention"""
    
    def __init__(self, vocab_size=256, d_model=512, nhead=8, num_layers=6,
                 dim_feedforward=2048, dropout=0.1, activation="gelu",
                 use_adaptive_attention=False):
        super().__init__()
        
        self.d_model = d_model
        
        # Better embedding initialization
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)
        
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Input normalization
        self.input_norm = nn.LayerNorm(d_model, eps=1e-5)
        
        # Create encoder layers with sparse attention
        self.layers = nn.ModuleList([
            SparseTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                layer_idx=i,
                use_adaptive=use_adaptive_attention
            ) for i in range(num_layers)
        ])
        
        # Final normalization
        self.norm = nn.LayerNorm(d_model, eps=1e-5)
        
        # Output projection with better initialization
        self.fc_out = nn.Linear(d_model, vocab_size)
        nn.init.normal_(self.fc_out.weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.fc_out.bias)
        
        # Gradient checkpointing state
        self.gradient_checkpointing = False
        
    def _layer_forward(self, layer, src, src_mask=None, src_key_padding_mask=None):
        """Helper function for gradient checkpointing"""
        def custom_forward(*inputs):
            return layer(*inputs)

        if self.gradient_checkpointing:
            return checkpoint(custom_forward, src, src_mask, src_key_padding_mask)
        return layer(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Single embedding scaling
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        # Input normalization
        src = self.input_norm(src)
        
        # Store attention weights for analysis
        attention_weights = []
        
        # Pass through encoder layers
        for layer in self.layers:
            if self.gradient_checkpointing:
                src, attn_weights = self._layer_forward(layer, src, src_mask, src_key_padding_mask)
            else:
                src, attn_weights = layer(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
            
            # Less aggressive intermediate clamping
            src = torch.clamp(src, min=-10.0, max=10.0)
            
            attention_weights.append(attn_weights)
        
        # Final normalization
        output = self.norm(src)
        
        # Project to vocabulary size without extra scaling
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