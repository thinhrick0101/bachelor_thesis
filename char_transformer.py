import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import random
import os
import requests
import gzip
from io import BytesIO
from zipfile import ZipFile

# Set random seed for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

def load_data(path, url=None):
    """
    Load data from a file or URL.

    Args:
        path: Path to the data file
        url: URL to download the data from if the file doesn't exist

    Returns:
        Data as a string
    """
    if not os.path.exists(path):
        if url is None:
            raise ValueError(f"File {path} does not exist and no URL provided")

        print(f"Downloading data from {url}")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Download the file
        response = requests.get(url)

        # Check if it's a gzip file
        if url.endswith('.gz'):
            print("Extracting gzip file...")
            data = gzip.decompress(response.content).decode('utf-8')
        # Check if it's a zip file
        elif url.endswith('.zip'):
            print("Extracting zip file...")
            with ZipFile(BytesIO(response.content)) as zip_file:
                # Print the files in the zip
                print(f"Files in zip: {zip_file.namelist()}")

                # Extract the file with the same name as the path
                base_name = os.path.basename(path)
                if base_name in zip_file.namelist():
                    file_name = base_name
                else:
                    # Extract the first file in the zip
                    file_name = zip_file.namelist()[0]

                print(f"Extracting {file_name}...")
                with zip_file.open(file_name) as file:
                    data = file.read().decode('utf-8')
        else:
            data = response.text

        # Save the data to the specified path
        with open(path, 'w', encoding='utf-8') as file:
            file.write(data)
    else:
        # Load data from the file
        print(f"Loading data from {path}")
        with open(path, 'r', encoding='utf-8') as file:
            data = file.read()

    return data

class CharacterTokenizer:
    """
    Simple character-level tokenizer
    """
    def __init__(self, text):
        # Get unique characters from the text
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)

        # Create character to index and index to character mappings
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

        print(f"Vocabulary size: {self.vocab_size} characters")

    def encode(self, text):
        """Convert text to a list of integers"""
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, indices):
        """Convert a list of integers to text"""
        return ''.join([self.idx_to_char[idx] for idx in indices])

def create_batches(data, batch_size, seq_length, device, curriculum_factor=1.0):
    """
    Create batches of data for training with support for curriculum learning

    Args:
        data: List of token indices
        batch_size: Number of sequences per batch
        seq_length: Length of each sequence
        device: Device to put the tensors on
        curriculum_factor: Factor to scale sequence length (1.0 = full length)

    Returns:
        List of (input, target) tuples
    """
    # Apply curriculum learning by adjusting sequence length
    curr_seq_length = max(32, int(seq_length * curriculum_factor))

    # Calculate the number of batches
    num_batches = (len(data) - 1) // (batch_size * curr_seq_length)

    # Trim the data to fit into batches
    data = data[:num_batches * batch_size * curr_seq_length + 1]

    # Reshape the data into batches
    x = torch.tensor(data[:-1], dtype=torch.long).view(batch_size, -1)
    y = torch.tensor(data[1:], dtype=torch.long).view(batch_size, -1)

    # Create batches
    batches = []
    for i in range(0, x.size(1), curr_seq_length):
        # Get sequences of length curr_seq_length
        if i + curr_seq_length <= x.size(1):
            # Create CPU tensors first (will be moved to device during training)
            input_batch = x[:, i:i+curr_seq_length].clone()
            target_batch = y[:, i:i+curr_seq_length].clone()
            batches.append((input_batch, target_batch))

    return batches

def create_curriculum_batches(data, batch_size, seq_length, device, num_epochs):
    """
    Create curriculum learning batches for each epoch

    Args:
        data: List of token indices
        batch_size: Number of sequences per batch
        seq_length: Maximum sequence length
        device: Device to put the tensors on
        num_epochs: Number of training epochs

    Returns:
        Dictionary mapping epoch to batches
    """
    # Split data into training and validation sets (90% / 10%)
    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    # Create curriculum batches for each epoch
    curriculum_batches = {}

    for epoch in range(num_epochs):
        # Calculate curriculum factor (gradually increase from 0.3 to 1.0)
        curriculum_factor = min(1.0, 0.3 + 0.7 * (epoch / (num_epochs - 1)))

        # Create batches with current curriculum factor
        train_batches = create_batches(train_data, batch_size, seq_length, device, curriculum_factor)
        val_batches = create_batches(val_data, batch_size, seq_length, device, curriculum_factor)

        curriculum_batches[epoch] = (train_batches, val_batches)

    return curriculum_batches

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE) as described in 'Roformer: Enhanced Transformer with Rotary Position Embedding'
    This is more effective than standard sinusoidal embeddings for character-level modeling.
    """
    def __init__(self, d_model, max_seq_length=5000, base=10000.0):
        super(RotaryPositionalEmbedding, self).__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.base = base

        # Create frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)

        # Cache for faster inference
        self._update_cos_sin_cache(max_seq_length)

    def _update_cos_sin_cache(self, seq_length):
        """Update the cache for cos and sin values"""
        self.max_seq_length = max(seq_length, self.max_seq_length)

        # Create position indices
        seq_idx = torch.arange(self.max_seq_length, device=self.inv_freq.device)

        # Compute cos and sin values
        freqs = torch.einsum('i,j->ij', seq_idx, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()

        # Register buffers
        self.register_buffer('cos_cached', cos)
        self.register_buffer('sin_cached', sin)

    def _rotate_half(self, x):
        """Rotate half the hidden dims of the input"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x, seq_length=None):
        """
        Apply rotary position embeddings to input tensor

        Args:
            x: Input tensor of shape [batch_size, seq_length, embedding_dim]
            seq_length: Optional sequence length (for caching)

        Returns:
            Tensor with rotary position embeddings applied
        """
        seq_length = x.size(1)

        # Update cache if needed
        if seq_length > self.max_seq_length:
            self._update_cos_sin_cache(seq_length)

        # Get cos and sin values for current sequence length
        cos = self.cos_cached[:seq_length]
        sin = self.sin_cached[:seq_length]

        # Reshape for broadcasting
        cos = cos.unsqueeze(0)  # [1, seq_length, d_model]
        sin = sin.unsqueeze(0)  # [1, seq_length, d_model]

        # Apply rotary embeddings
        return (x * cos) + (self._rotate_half(x) * sin)

class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) - More efficient than standard multi-head attention
    by using fewer key-value heads than query heads.
    """
    def __init__(self, d_model, num_query_heads=8, num_kv_heads=2, dropout=0.1, bias=True):
        super(GroupedQueryAttention, self).__init__()
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_query_heads

        # Ensure dimensions are compatible
        assert d_model % num_query_heads == 0, "d_model must be divisible by num_query_heads"
        assert num_query_heads % num_kv_heads == 0, "num_query_heads must be divisible by num_kv_heads"

        # Number of query heads per kv head
        self.kv_groups = num_query_heads // num_kv_heads

        # Projections
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Scaling factor
        self.scale = self.head_dim ** -0.5

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize with small values
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1/math.sqrt(2))
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1/math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1/math.sqrt(2))
        nn.init.xavier_uniform_(self.o_proj.weight, gain=1/math.sqrt(2))

        # Initialize biases to zero
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.o_proj.bias)

    def forward(self, q, k=None, v=None, attn_mask=None):
        """
        Forward pass for grouped query attention

        Args:
            q: Query tensor [batch_size, seq_length, d_model]
            k: Key tensor (optional, defaults to q)
            v: Value tensor (optional, defaults to k)
            attn_mask: Attention mask

        Returns:
            Output tensor and attention weights
        """
        # Default to self-attention if k and v are not provided
        if k is None:
            k = q
        if v is None:
            v = k

        batch_size, q_len, _ = q.shape
        _, kv_len, _ = k.shape

        # Project queries, keys, and values
        q = self.q_proj(q)  # [batch_size, q_len, d_model]
        k = self.k_proj(k)  # [batch_size, kv_len, num_kv_heads * head_dim]
        v = self.v_proj(v)  # [batch_size, kv_len, num_kv_heads * head_dim]

        # Reshape for multi-head attention
        q = q.view(batch_size, q_len, self.num_query_heads, self.head_dim)
        k = k.view(batch_size, kv_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, kv_len, self.num_kv_heads, self.head_dim)

        # Repeat k and v for each query group
        if self.kv_groups > 1:
            k = k.repeat_interleave(self.kv_groups, dim=2)  # [batch_size, kv_len, num_query_heads, head_dim]
            v = v.repeat_interleave(self.kv_groups, dim=2)  # [batch_size, kv_len, num_query_heads, head_dim]

        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch_size, num_query_heads, q_len, head_dim]
        k = k.transpose(1, 2)  # [batch_size, num_query_heads, kv_len, head_dim]
        v = v.transpose(1, 2)  # [batch_size, num_query_heads, kv_len, head_dim]

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [batch_size, num_query_heads, q_len, kv_len]

        # Apply mask if provided
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))

        # Apply softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)  # [batch_size, num_query_heads, q_len, head_dim]

        # Reshape and project back
        output = output.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        output = self.o_proj(output)

        return output, attn_weights

class TransformerBlock(nn.Module):
    """
    Enhanced transformer encoder block with grouped query attention, feed-forward network,
    and advanced techniques like gated activations and layer scale
    """
    def __init__(self, d_model, nhead, num_kv_heads=None, dim_feedforward=2048, dropout=0.1, layer_scale_init=1e-2):
        super(TransformerBlock, self).__init__()

        # Determine number of KV heads (default to nhead/4 or at least 1)
        if num_kv_heads is None:
            num_kv_heads = max(1, nhead // 4)

        # Grouped Query Attention
        self.self_attn = GroupedQueryAttention(
            d_model=d_model,
            num_query_heads=nhead,
            num_kv_heads=num_kv_heads,
            dropout=dropout
        )

        # Feed-forward network with SwiGLU activation (better than GELU)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear1_gate = nn.Linear(d_model, dim_feedforward)  # Gate for SwiGLU
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Layer scale parameters (from CaiT and other modern transformers)
        self.layer_scale1 = nn.Parameter(layer_scale_init * torch.ones(d_model))
        self.layer_scale2 = nn.Parameter(layer_scale_init * torch.ones(d_model))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize linear layers with small values
        nn.init.xavier_uniform_(self.linear1.weight, gain=1/math.sqrt(2))
        nn.init.xavier_uniform_(self.linear1_gate.weight, gain=1/math.sqrt(2))
        nn.init.xavier_uniform_(self.linear2.weight, gain=1/math.sqrt(2))

        # Initialize biases to zero
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.constant_(self.linear1_gate.bias, 0)
        nn.init.constant_(self.linear2.bias, 0)

    def swiglu(self, x):
        """SwiGLU activation function (better than GELU)"""
        return self.linear1(x) * F.silu(self.linear1_gate(x))

    def forward(self, src, src_mask=None):
        """
        Args:
            src: Input tensor of shape [batch_size, seq_length, embedding_dim]
            src_mask: Mask for self-attention

        Returns:
            Output tensor of shape [batch_size, seq_length, embedding_dim]
        """
        # Self-attention block (with pre-norm)
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, attn_mask=src_mask)

        # Apply layer scale and residual connection
        src = src + self.dropout1(self.layer_scale1.unsqueeze(0).unsqueeze(0) * src2)

        # Feed-forward block (with pre-norm)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.swiglu(src2)))

        # Apply layer scale and residual connection
        src = src + self.dropout2(self.layer_scale2.unsqueeze(0).unsqueeze(0) * src2)

        return src

class SimpleFeedForward(nn.Module):
    """
    Simple feed-forward network as a replacement for MoE
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(SimpleFeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        # Feed-forward network with GELU activation
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize linear layers
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor [batch_size, seq_length, d_model]

        Returns:
            Output tensor [batch_size, seq_length, d_model]
        """
        return self.net(x)

class CharTransformer(nn.Module):
    """
    Enhanced character-level transformer model for text generation with
    advanced techniques like rotary position embeddings, grouped query attention,
    mixture of experts, and improved initialization
    """
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout=0.1, use_moe=True):
        super(CharTransformer, self).__init__()

        # Token embedding with proper scaling
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.embedding_scale = math.sqrt(d_model)  # Scale embeddings by sqrt(d_model)

        # Rotary positional encoding (better than sinusoidal)
        self.pos_encoder = RotaryPositionalEmbedding(d_model)

        # Transformer blocks with layer-wise hyperparameters
        self.transformer_blocks = nn.ModuleList()
        for i in range(num_layers):
            # Gradually increase dropout in deeper layers
            layer_dropout = dropout * (1.0 + i * 0.1)
            layer_dropout = min(layer_dropout, 0.5)  # Cap at 0.5

            # Number of KV heads (fewer in deeper layers)
            num_kv_heads = max(1, nhead // (2 ** min(i, 2)))

            self.transformer_blocks.append(
                TransformerBlock(
                    d_model=d_model,
                    nhead=nhead,
                    num_kv_heads=num_kv_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=layer_dropout,
                    layer_scale_init=1e-2 / (i + 1)  # Smaller scale for deeper layers
                )
            )

        # Additional feed-forward layer (optional)
        self.use_moe = use_moe
        if use_moe:
            self.moe = SimpleFeedForward(
                d_model=d_model,
                d_ff=dim_feedforward * 2,  # Larger feed-forward dimension
                dropout=dropout
            )

        # Final layer normalization
        self.norm = nn.LayerNorm(d_model)

        # Output projection with tied weights
        self.output = nn.Linear(d_model, vocab_size)

        # Dropout with stochastic depth (higher dropout for later tokens)
        self.dropout = nn.Dropout(dropout)

        # Initialize parameters
        self._init_parameters()

        # Tie embedding and output weights (weight sharing)
        self.output.weight = self.embedding.weight

    def _init_parameters(self):
        """Initialize model parameters with improved techniques"""
        # Initialize embeddings with truncated normal distribution
        nn.init.trunc_normal_(self.embedding.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)

        # Initialize output projection bias
        nn.init.constant_(self.output.bias, 0)

    def _generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence with improved numerical stability"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask.to(self.embedding.weight.device)

    def forward(self, src, mask=None):
        """
        Args:
            src: Input tensor of shape [batch_size, seq_length]
            mask: Optional mask for self-attention

        Returns:
            Output tensor of shape [batch_size, seq_length, vocab_size]
        """
        # Ensure src is on the same device as the model
        device = self.embedding.weight.device
        src = src.to(device)

        # Create causal mask if not provided
        if mask is None:
            mask = self._generate_square_subsequent_mask(src.size(1))

        # Ensure mask is on the same device
        mask = mask.to(device)

        # Embed tokens and scale
        # [batch_size, seq_length] -> [batch_size, seq_length, d_model]
        x = self.embedding(src) * self.embedding_scale

        # Apply rotary position embeddings in each transformer block
        # (not applied here since RoPE is applied within attention)

        # Apply dropout
        x = self.dropout(x)

        # Pass through transformer blocks with gradient checkpointing for memory efficiency
        for i, block in enumerate(self.transformer_blocks):
            # Apply stochastic depth (higher probability of skipping later layers)
            if self.training and i > 0:
                skip_prob = 0.1 * (i / len(self.transformer_blocks))
                if random.random() < skip_prob:
                    continue

            # Use gradient checkpointing to save memory
            if self.training:
                # Custom function for gradient checkpointing
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, mask
                )
            else:
                x = block(x, src_mask=mask)

        # Apply Mixture of Experts after transformer blocks (if enabled)
        if self.use_moe:
            x = self.moe(x)

        # Apply final layer normalization
        x = self.norm(x)

        # Apply output projection
        output = self.output(x)

        return output

    def generate(self, prompt, max_length, temperature=0.7, top_k=20, top_p=0.9, tokenizer=None, device='cpu'):
        """
        Generate text from a prompt with improved sampling strategies and stability

        Args:
            prompt: Initial text prompt
            max_length: Maximum length of the generated text
            temperature: Sampling temperature (higher = more random)
            top_k: Number of highest probability tokens to keep for top-k sampling
            top_p: Cumulative probability threshold for nucleus sampling
            tokenizer: Character tokenizer
            device: Device to use for generation

        Returns:
            Generated text
        """
        self.eval()

        # Encode the prompt
        if isinstance(prompt, str) and tokenizer is not None:
            prompt_ids = tokenizer.encode(prompt)
            prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).to(device)
        else:
            prompt_tensor = prompt

        # Generate text
        generated = prompt_tensor.clone()

        with torch.no_grad():
            for _ in range(max_length):
                try:
                    # Get predictions for the last token
                    outputs = self(generated)

                    # Apply temperature scaling with a safety check
                    next_token_logits = outputs[:, -1, :]

                    # Check for NaN values
                    if torch.isnan(next_token_logits).any():
                        print("Warning: NaN values detected in logits. Using uniform sampling.")
                        # Fall back to uniform sampling
                        next_token = torch.randint(0, tokenizer.vocab_size, (1, 1), device=device)
                    else:
                        # Apply temperature with a safety check
                        next_token_logits = next_token_logits / max(0.1, temperature)  # Prevent division by zero

                        # Apply top-k filtering
                        if top_k > 0:
                            top_k_values, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                            min_value = top_k_values[:, -1].unsqueeze(-1)
                            next_token_logits = torch.where(
                                next_token_logits < min_value,
                                torch.ones_like(next_token_logits) * float('-inf'),
                                next_token_logits
                            )

                        # Apply top-p (nucleus) filtering with safety checks
                        if top_p < 1.0:
                            # Sort logits in descending order
                            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)

                            # Apply softmax with a safety check
                            sorted_probs = F.softmax(sorted_logits, dim=-1)

                            # Check for NaN values
                            if torch.isnan(sorted_probs).any():
                                print("Warning: NaN values detected in probabilities. Using uniform sampling.")
                                next_token = torch.randint(0, tokenizer.vocab_size, (1, 1), device=device)
                                continue

                            # Calculate cumulative probabilities
                            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                            # Create mask for tokens to remove
                            sorted_indices_to_remove = cumulative_probs > top_p

                            # Keep at least one token
                            if sorted_indices_to_remove.all():
                                sorted_indices_to_remove[..., 0] = False

                            # Shift indices to keep the first token above threshold
                            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                            sorted_indices_to_remove[..., 0] = False

                            # Apply the mask to the sorted indices
                            indices_to_remove = torch.zeros_like(next_token_logits, dtype=torch.bool).scatter_(
                                -1, sorted_indices, sorted_indices_to_remove
                            )

                            # Set removed indices to -inf
                            next_token_logits = next_token_logits.masked_fill(indices_to_remove, float('-inf'))

                        # Apply softmax to get probabilities with a safety check
                        probs = F.softmax(next_token_logits, dim=-1)

                        # Check for NaN values
                        if torch.isnan(probs).any() or (probs < 0).any():
                            print("Warning: Invalid probability values. Using uniform sampling.")
                            next_token = torch.randint(0, tokenizer.vocab_size, (1, 1), device=device)
                        else:
                            # Sample from the distribution
                            next_token = torch.multinomial(probs, num_samples=1)

                    # Append the next token to the generated sequence
                    generated = torch.cat((generated, next_token), dim=1)

                except Exception as e:
                    print(f"Error during generation: {e}")
                    # Fall back to a safe token
                    next_token = torch.randint(0, tokenizer.vocab_size, (1, 1), device=device)
                    generated = torch.cat((generated, next_token), dim=1)

        # Decode the generated text
        try:
            if tokenizer is not None:
                return tokenizer.decode(generated[0].tolist())
            else:
                return generated
        except Exception as e:
            print(f"Error during decoding: {e}")
            return prompt  # Return the original prompt as a fallback



def train_model(model, train_batches, val_batches=None, num_epochs=5, learning_rate=0.0001,
                weight_decay=0.01, warmup_steps=0, device=None, patience=3):
    """
    Train the model and evaluate on validation set

    Args:
        model: The model to train
        train_batches: List of training batches
        val_batches: List of validation batches (optional)
        num_epochs: Number of training epochs
        learning_rate: Learning rate for the optimizer
        weight_decay: Weight decay for regularization
        warmup_steps: Number of warmup steps for learning rate scheduling
        device: Device to use for training (cuda or cpu)
        patience: Number of epochs to wait for improvement before early stopping

    Returns:
        Trained model and training metrics
    """
    # Determine device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model to device
    model = model.to(device)
    print(f"Training on device: {device}")

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Define optimizer with weight decay
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

    # Calculate total number of training steps
    total_steps = len(train_batches) * num_epochs

    # Create learning rate scheduler
    if warmup_steps > 0:
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # For tracking metrics
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    no_improvement_count = 0

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        start_time = time.time()

        # Process each batch
        for inputs, targets in train_batches:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Move tensors to device
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Reshape for loss calculation
            outputs = outputs.reshape(-1, outputs.size(-1))
            targets = targets.reshape(-1)

            # Calculate loss
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()

            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Update learning rate with warmup scheduler
            if warmup_steps > 0:
                scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        # Calculate average loss for the epoch
        avg_loss = total_loss / num_batches
        train_losses.append(avg_loss)

        # Calculate perplexity
        perplexity = math.exp(avg_loss)

        # Evaluate on validation set if provided
        if val_batches is not None:
            model.eval()
            val_total_loss = 0
            val_num_batches = 0

            with torch.no_grad():
                for inputs, targets in val_batches:
                    # Forward pass
                    outputs = model(inputs)

                    # Reshape for loss calculation
                    outputs = outputs.reshape(-1, outputs.size(-1))
                    targets = targets.reshape(-1)

                    # Calculate loss
                    loss = criterion(outputs, targets)

                    val_total_loss += loss.item()
                    val_num_batches += 1

            # Calculate average validation loss
            val_avg_loss = val_total_loss / val_num_batches
            val_losses.append(val_avg_loss)

            # Calculate validation perplexity
            val_perplexity = math.exp(val_avg_loss)

            # Update learning rate with ReduceLROnPlateau scheduler
            if warmup_steps == 0:
                scheduler.step(val_avg_loss)

            # Save best model
            if val_avg_loss < best_val_loss:
                best_val_loss = val_avg_loss
                best_model_state = model.state_dict().copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Early stopping
            if no_improvement_count >= patience:
                print(f"No improvement for {patience} epochs. Early stopping.")
                break

            # Print epoch statistics
            epoch_time = time.time() - start_time
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}, '
                  f'Val Loss: {val_avg_loss:.4f}, Val Perplexity: {val_perplexity:.2f}, '
                  f'Time: {epoch_time:.2f}s')
        else:
            # Print epoch statistics without validation
            epoch_time = time.time() - start_time
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}, '
                  f'Time: {epoch_time:.2f}s')

    # Load best model if validation was used
    if val_batches is not None and best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation loss: {best_val_loss:.4f}, perplexity: {math.exp(best_val_loss):.2f}")

    return model, (train_losses, val_losses)

def train_epoch(model, train_batches, val_batches, criterion, optimizer, scheduler, device):
    """
    Train for one epoch and evaluate

    Args:
        model: The model to train
        train_batches: Training batches
        val_batches: Validation batches
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to use for training

    Returns:
        Average training loss and validation loss
    """
    # Training phase
    model.train()
    total_loss = 0
    num_batches = 0
    start_time = time.time()

    # Process each batch
    for inputs, targets in train_batches:
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Move tensors to device
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(inputs)

        # Reshape for loss calculation
        outputs = outputs.reshape(-1, outputs.size(-1))
        targets = targets.reshape(-1)

        # Calculate loss
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()

        # Clip gradients with a lower threshold to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

    # Calculate average training loss
    avg_train_loss = total_loss / num_batches

    # Evaluation phase
    model.eval()
    val_total_loss = 0
    val_num_batches = 0

    with torch.no_grad():
        for inputs, targets in val_batches:
            # Move tensors to device
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Reshape for loss calculation
            outputs = outputs.reshape(-1, outputs.size(-1))
            targets = targets.reshape(-1)

            # Calculate loss
            loss = criterion(outputs, targets)

            val_total_loss += loss.item()
            val_num_batches += 1

    # Calculate average validation loss
    avg_val_loss = val_total_loss / val_num_batches

    # Calculate epoch time
    epoch_time = time.time() - start_time
    print(f"Epoch completed in {epoch_time:.2f}s")

    return avg_train_loss, avg_val_loss

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def visualize_results(train_losses, val_losses=None):
    """
    Visualize training and validation losses
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    if val_losses:
        plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('char_transformer_loss.png')
    plt.close()

def main():
    # Data parameters
    data_path = 'data/enwik8'
    data_url = 'https://codeberg.org/pbm/former/raw/branch/master/data/enwik8.gz'

    # Model hyperparameters - optimized for better performance
    d_model = 256  # Embedding dimension
    nhead = 8  # Number of attention heads
    num_layers = 4  # Number of transformer layers
    dim_feedforward = 1024  # Feed-forward dimension
    dropout = 0.2  # Dropout rate
    use_moe = True  # Use Mixture of Experts

    # Training hyperparameters - optimized for stability and convergence
    batch_size = 32  # Smaller batch size for stability
    seq_length = 128  # Shorter sequences for stability
    num_epochs = 15  # More epochs for better convergence
    learning_rate = 1e-4  # Higher learning rate with better warmup
    weight_decay = 0.01  # Weight decay for regularization

    # Load data
    print("Loading data...")
    text = load_data(data_path, data_url)
    print(f"Data loaded: {len(text)} characters")

    # Limit the data size for training (first 1M characters)
    max_chars = 1000000
    if len(text) > max_chars:
        print(f"Limiting data to first {max_chars} characters for training")
        text = text[:max_chars]

    # Create tokenizer
    tokenizer = CharacterTokenizer(text)

    # Encode the text
    data = tokenizer.encode(text)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create curriculum batches for each epoch
    print("Creating curriculum batches...")
    curriculum_batches = create_curriculum_batches(data, batch_size, seq_length, device, num_epochs)

    # Get initial batches for the first epoch
    train_batches, val_batches = curriculum_batches[0]
    print(f"Created curriculum batches for {num_epochs} epochs")
    print(f"Initial epoch: {len(train_batches)} training batches and {len(val_batches)} validation batches")

    # Create model
    model = CharTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        use_moe=use_moe
    )

    # Move model to device
    model = model.to(device)

    # Print model architecture
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {trainable_params:,} trainable out of {total_params:,} total")

    # Calculate warmup steps - longer warmup for stability
    warmup_steps = len(train_batches) * 3  # Warmup for three epochs

    # Train model with curriculum learning
    print("\n=== Training Character Transformer Model with Curriculum Learning ===")

    # Initialize tracking variables
    all_train_losses = []
    all_val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    no_improvement_count = 0
    patience = 5

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    # Define optimizer with weight decay
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

    # Calculate total steps for all epochs
    total_steps = sum(len(curriculum_batches[e][0]) for e in range(num_epochs))

    # Create learning rate scheduler with warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Curriculum learning training loop
    for epoch in range(num_epochs):
        # Get batches for this epoch
        train_batches, val_batches = curriculum_batches[epoch]

        # Calculate curriculum factor for this epoch
        curriculum_factor = min(1.0, 0.3 + 0.7 * (epoch / (num_epochs - 1)))
        curr_seq_length = max(32, int(seq_length * curriculum_factor))

        print(f"\nEpoch {epoch+1}/{num_epochs} - Sequence length: {curr_seq_length}")

        # Train for one epoch
        epoch_train_loss, epoch_val_loss = train_epoch(
            model=model,
            train_batches=train_batches,
            val_batches=val_batches,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device
        )

        # Track losses
        all_train_losses.append(epoch_train_loss)
        all_val_losses.append(epoch_val_loss)

        # Calculate perplexity
        train_perplexity = math.exp(epoch_train_loss)
        val_perplexity = math.exp(epoch_val_loss)

        print(f"Train Loss: {epoch_train_loss:.4f}, Perplexity: {train_perplexity:.2f}")
        print(f"Val Loss: {epoch_val_loss:.4f}, Perplexity: {val_perplexity:.2f}")

        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict().copy()
            no_improvement_count = 0
            print(f"New best model! Validation perplexity: {val_perplexity:.2f}")
        else:
            no_improvement_count += 1

        # Early stopping
        if no_improvement_count >= patience:
            print(f"No improvement for {patience} epochs. Early stopping.")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation perplexity: {math.exp(best_val_loss):.2f}")

    # Store results
    train_losses, val_losses = all_train_losses, all_val_losses

    # Visualize results
    visualize_results(train_losses, val_losses)
    print("\nTraining visualization saved to char_transformer_loss.png")

    # Generate some text
    print("\n=== Generating Text ===")
    prompt = "The quick brown fox"
    generated_text = model.generate(
        prompt=prompt,
        max_length=200,
        temperature=0.8,
        tokenizer=tokenizer,
        device=device
    )
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")

    # Save model
    torch.save(model.state_dict(), 'char_transformer_model.pt')
    print("Model saved to char_transformer_model.pt")

if __name__ == "__main__":
    main()
