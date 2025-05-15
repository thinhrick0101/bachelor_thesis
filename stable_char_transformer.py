import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import math
import random
import os
import gzip
import urllib.request
from torch.cuda.amp import autocast, GradScaler  # For mixed precision training
from torch.utils.checkpoint import checkpoint  # For gradient checkpointing
from tokenizers import Tokenizer  # Import Hugging Face tokenizer
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
import argparse

def load_data(data_path, data_url=None):
    """
    Load text data from file or download if not available
    """
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    # Download data if not available
    if not os.path.exists(data_path) and data_url:
        print(f"Downloading data from {data_url}")
        urllib.request.urlretrieve(data_url, data_path + '.gz')

        # Decompress .gz file
        with gzip.open(data_path + '.gz', 'rb') as f_in:
            with open(data_path, 'wb') as f_out:
                f_out.write(f_in.read())

    # Load data
    print(f"Loading data from {data_path}")
    with open(data_path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()

    return text

def create_batches(data, batch_size, seq_length):
    """
    Create batches of data for training

    Args:
        data: List of token indices
        batch_size: Number of sequences per batch
        seq_length: Length of each sequence

    Returns:
        List of (input, target) tuples
    """
    # Calculate the number of batches
    num_batches = (len(data) - 1) // (batch_size * seq_length)

    # Trim the data to fit into batches
    data = data[:num_batches * batch_size * seq_length + 1]

    # Reshape the data into batches
    x = torch.tensor(data[:-1], dtype=torch.long).view(batch_size, -1)
    y = torch.tensor(data[1:], dtype=torch.long).view(batch_size, -1)

    # Create batches
    batches = []
    for i in range(0, x.size(1), seq_length):
        # Get sequences of length seq_length
        if i + seq_length <= x.size(1):
            input_batch = x[:, i:i+seq_length].clone()
            target_batch = y[:, i:i+seq_length].clone()
            batches.append((input_batch, target_batch))

    return batches

class ImprovedPositionalEncoding(nn.Module):
    """
    Improved positional encoding with learnable parameters and better initialization
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(ImprovedPositionalEncoding, self).__init__()

        # Create base positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register buffer (persistent state)
        self.register_buffer('pe', pe)

        # Learnable scaling factor for positional encoding
        self.alpha = nn.Parameter(torch.ones(1))

        # Learnable position-wise feed-forward layer
        self.position_ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout)
        )

        # Layer normalization
        self.norm = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for the learnable components"""
        for module in self.position_ff.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_length, embedding_dim]

        Returns:
            Positional encoding added to input embeddings
        """
        # Get base positional encoding
        pos_enc = self.pe[:x.size(1), :].unsqueeze(0)

        # Apply learnable scaling
        pos_enc = self.alpha * pos_enc

        # Add positional encoding to input embeddings
        x = x + pos_enc

        # Apply position-wise feed-forward layer
        x = x + self.dropout(self.position_ff(self.norm(x)))

        return x

class EnhancedTransformerBlock(nn.Module):
    """
    Enhanced transformer encoder block with gradient checkpointing, improved attention,
    and better regularization techniques
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 attention_dropout=0.1, activation_dropout=0.1, use_checkpoint=True):
        super(EnhancedTransformerBlock, self).__init__()

        # Multi-head self-attention with scaled dot-product attention
        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=attention_dropout,  # Separate dropout for attention
            batch_first=True
        )

        # Improved feed-forward network with SwiGLU activation (better than GELU)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward * 2),  # Double size for SwiGLU
            nn.SiLU(),  # SiLU (Swish) activation
            nn.Dropout(activation_dropout),  # Separate dropout for activations
            nn.Linear(dim_feedforward * 2, d_model),
            nn.Dropout(dropout)
        )

        # Layer normalization with better epsilon
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Residual scaling factors (learnable)
        self.gamma1 = nn.Parameter(torch.ones(1))
        self.gamma2 = nn.Parameter(torch.ones(1))

        # Gradient checkpointing flag
        self.use_checkpoint = use_checkpoint

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with improved techniques"""
        # Initialize feed-forward network
        for module in self.feed_forward.modules():
            if isinstance(module, nn.Linear):
                # Use truncated normal distribution for better stability
                nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _attention_block(self, src, src_mask=None):
        """Self-attention block with pre-norm"""
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2, attn_mask=src_mask, need_weights=False)
        return self.dropout1(self.gamma1 * src2)

    def _ff_block(self, src):
        """Feed-forward block with pre-norm"""
        src2 = self.norm2(src)
        src2 = self.feed_forward(src2)
        return self.dropout2(self.gamma2 * src2)

    def forward(self, src, src_mask=None):
        """
        Args:
            src: Input tensor of shape [batch_size, seq_length, embedding_dim]
            src_mask: Mask for self-attention

        Returns:
            Output tensor of shape [batch_size, seq_length, embedding_dim]
        """
        # Apply gradient checkpointing if enabled (saves memory during training)
        if self.use_checkpoint and self.training:
            # Self-attention block with gradient checkpointing
            src = src + checkpoint(
                lambda x, mask: self._attention_block(x, mask),
                src, src_mask,
                use_reentrant=False  # Add explicit use_reentrant parameter
            )

            # Feed-forward block with gradient checkpointing
            src = src + checkpoint(
                self._ff_block,
                src,
                use_reentrant=False  # Add explicit use_reentrant parameter
            )
        else:
            # Self-attention block (with pre-norm)
            src = src + self._attention_block(src, src_mask)

            # Feed-forward block (with pre-norm)
            src = src + self._ff_block(src)

        return src

class EnhancedCharTransformer(nn.Module):
    """
    Enhanced character-level transformer model for text generation
    with improved positional encoding, enhanced transformer blocks, token-level dropout,
    sophisticated initialization, and gradient checkpointing for memory efficiency
    """
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward,
                 dropout=0.1, attention_dropout=0.1, activation_dropout=0.1,
                 token_dropout=0.05, use_checkpoint=True, stochastic_depth_prob=0.1):
        super(EnhancedCharTransformer, self).__init__()

        # Token embedding with weight tying preparation
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Embedding scale factor (optional but helps training)
        self.embed_scale = math.sqrt(d_model)

        # Improved positional encoding
        self.pos_encoder = ImprovedPositionalEncoding(d_model, dropout=dropout)

        # Enhanced transformer blocks with progressive dropout
        self.transformer_blocks = nn.ModuleList()
        for i in range(num_layers):
            # Gradually increase dropout in deeper layers
            layer_dropout = dropout * (1.0 + i * 0.1)
            layer_dropout = min(layer_dropout, 0.5)  # Cap at 0.5

            # Gradually increase attention dropout in deeper layers
            layer_attn_dropout = attention_dropout * (1.0 + i * 0.05)
            layer_attn_dropout = min(layer_attn_dropout, 0.4)  # Cap at 0.4

            self.transformer_blocks.append(
                EnhancedTransformerBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=layer_dropout,
                    attention_dropout=layer_attn_dropout,
                    activation_dropout=activation_dropout,
                    use_checkpoint=use_checkpoint
                )
            )

        # Final layer normalization
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

        # Output projection (weight tied with embedding)
        self.output = nn.Linear(d_model, vocab_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Token-level dropout for better generalization
        self.token_dropout = token_dropout

        # Stochastic depth probability
        self.stochastic_depth_prob = stochastic_depth_prob

        # Initialize parameters
        self._init_parameters()

        # Tie weights between embedding and output projection
        self.output.weight = self.embedding.weight

    def _init_parameters(self):
        """Initialize model parameters with improved techniques"""
        # Initialize embeddings
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)

        # Initialize output projection bias
        if self.output.bias is not None:
            nn.init.zeros_(self.output.bias)

    def _generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, mask=None):
        """
        Args:
            src: Input tensor of shape [batch_size, seq_length]
            mask: Optional mask for self-attention

        Returns:
            Output tensor of shape [batch_size, seq_length, vocab_size]
        """
        # Create causal mask if not provided
        if mask is None:
            mask = self._generate_square_subsequent_mask(src.size(1)).to(src.device)

        # Apply token-level dropout during training
        if self.training and self.token_dropout > 0:
            # Create a random mask for token dropout
            token_mask = torch.bernoulli(
                torch.full_like(src, 1 - self.token_dropout, dtype=torch.float)
            ).bool()

            # Replace dropped tokens with a special token (0 for simplicity)
            # This simulates missing or corrupted tokens
            src = torch.where(token_mask, src, torch.zeros_like(src))

        # Embed tokens and scale
        # [batch_size, seq_length] -> [batch_size, seq_length, d_model]
        x = self.embedding(src) * self.embed_scale

        # Add positional encoding
        x = self.pos_encoder(x)

        # Apply dropout
        x = self.dropout(x)

        # Pass through transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            # Apply stochastic depth (higher probability of skipping later layers)
            if self.training and i > 0:
                skip_prob = self.stochastic_depth_prob * (i / len(self.transformer_blocks))
                if random.random() < skip_prob:
                    continue

            x = block(x, src_mask=mask)

        # Apply final layer normalization
        x = self.norm(x)

        # Apply output projection
        output = self.output(x)

        return output

    def generate(self, prompt, max_length, temperature=0.7, top_k=20, top_p=0.9,
                repetition_penalty=1.2, tokenizer=None, device='cpu'):
        """
        Generate text from a prompt with improved sampling strategies and stability

        Args:
            prompt: Initial text prompt
            max_length: Maximum length of the generated sequence (in tokens)
            temperature: Sampling temperature (higher = more random)
            top_k: Number of highest probability tokens to keep for top-k sampling
            top_p: Cumulative probability threshold for nucleus sampling
            repetition_penalty: Penalty for repeating tokens (1.0 = no penalty)
            tokenizer: Hugging Face tokenizer instance
            device: Device to use for generation

        Returns:
            Generated text
        """
        self.eval()

        # Encode the prompt
        if isinstance(prompt, str) and tokenizer is not None:
            encoded = tokenizer.encode(prompt)
            prompt_tensor = torch.tensor(encoded.ids, dtype=torch.long).unsqueeze(0).to(device)
        else:
            prompt_tensor = prompt

        # Generate text
        generated = prompt_tensor.clone()

        # Keep track of past tokens for repetition penalty
        past_tokens = set()
        for token in generated[0].tolist():
            past_tokens.add(token)

        with torch.no_grad():
            for _ in range(max_length):
                try:
                    # Get predictions for the last token
                    # Use a sliding window approach for long sequences to save memory
                    if generated.size(1) > 1024:
                        # Use only the last 1024 tokens for context
                        context = generated[:, -1024:]
                    else:
                        context = generated

                    outputs = self(context)

                    # Apply temperature scaling with a safety check
                    next_token_logits = outputs[:, -1, :].clone()

                    # Check for NaN values
                    if torch.isnan(next_token_logits).any():
                        print("Warning: NaN values detected in logits. Using uniform sampling.")
                        next_token = torch.randint(0, tokenizer.get_vocab_size(), (1, 1), device=device)
                    else:
                        # Apply repetition penalty
                        if repetition_penalty > 1.0:
                            for token_id in past_tokens:
                                if token_id < next_token_logits.size(-1):
                                    next_token_logits[:, token_id] /= repetition_penalty

                        # Apply temperature with a safety check
                        next_token_logits = next_token_logits / max(0.1, temperature)

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
                            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                            sorted_probs = F.softmax(sorted_logits, dim=-1)

                            # Check for NaN values
                            if torch.isnan(sorted_probs).any():
                                print("Warning: NaN values detected in probabilities. Using uniform sampling.")
                                next_token = torch.randint(0, tokenizer.get_vocab_size(), (1, 1), device=device)
                                continue

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
                            next_token = torch.randint(0, tokenizer.get_vocab_size(), (1, 1), device=device)
                        else:
                            # Sample from the distribution
                            next_token = torch.multinomial(probs, num_samples=1)

                    # Add the new token to past tokens for repetition penalty
                    past_tokens.add(next_token.item())

                    # Check if we've generated an end token
                    if next_token.item() == tokenizer.token_to_id("[SEP]"):
                        break

                    # Append the next token to the generated sequence
                    generated = torch.cat((generated, next_token), dim=1)

                except Exception as e:
                    print(f"Error during generation: {e}")
                    # Fall back to a safe token
                    next_token = torch.randint(0, tokenizer.get_vocab_size(), (1, 1), device=device)
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

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=0.0, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to min_lr, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (1.0 - min_lr) * cosine_decay

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def setup_distributed(args):
    """Setup distributed training"""
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=f'tcp://{args.master_addr}:{args.master_port}',
            world_size=args.world_size,
            rank=args.local_rank
        )
        print(f"Initialized process {args.local_rank} of {args.world_size}")

def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

def create_distributed_batches(data, batch_size, seq_length, rank, world_size):
    """Create batches for distributed training"""
    # Calculate total number of sequences that can fit in batches
    total_seq = (len(data) - 1) // seq_length
    # Calculate sequences per batch
    seqs_per_batch = total_seq // batch_size
    # Ensure we have enough complete batches
    usable_seqs = seqs_per_batch * batch_size
    
    # Trim data to fit complete sequences
    usable_length = usable_seqs * seq_length
    data = data[:usable_length + 1]  # +1 for target shifting
    
    # Create tensor from data
    data_tensor = torch.tensor(data, dtype=torch.long)
    
    # Calculate size for each rank
    per_rank_seqs = usable_seqs // world_size
    seqs_remainder = usable_seqs % world_size
    
    # Adjust sequences for this rank
    if rank < seqs_remainder:
        rank_seqs = per_rank_seqs + 1
        start_seq = rank * (per_rank_seqs + 1)
    else:
        rank_seqs = per_rank_seqs
        start_seq = (rank * per_rank_seqs) + seqs_remainder
    
    # Calculate data indices for this rank
    start_idx = start_seq * seq_length
    end_idx = start_idx + (rank_seqs * seq_length) + 1  # +1 for target shifting
    
    # Get local data
    local_data = data_tensor[start_idx:end_idx]
    
    # Create batches
    batches = []
    local_seqs = rank_seqs
    
    # Reshape data into batches
    if local_seqs > 0:
        # Calculate local batch size to ensure even division
        local_batch_size = min(batch_size, local_seqs)
        while local_seqs % local_batch_size != 0:
            local_batch_size -= 1
        
        if local_batch_size > 0:
            # Reshape data into [batch_size, sequence_length]
            x = local_data[:-1].view(local_batch_size, -1)
            y = local_data[1:].view(local_batch_size, -1)
            
            # Create batches
            for i in range(0, x.size(1), seq_length):
                if i + seq_length <= x.size(1):
                    input_batch = x[:, i:i+seq_length].clone()
                    target_batch = y[:, i:i+seq_length].clone()
                    batches.append((input_batch, target_batch))
    
    return batches

def train_distributed_model(model, train_batches, val_batches=None, num_epochs=5, learning_rate=0.0001,
                weight_decay=0.01, warmup_steps=0, min_lr=0.0, device=None, patience=3, 
                label_smoothing=0.0, gradient_accumulation_steps=1, use_mixed_precision=True,
                use_cosine_schedule=False, local_rank=-1):
    """Train the model in a distributed setting"""
    # Set up distributed training
    is_distributed = local_rank != -1
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # Add progress tracking variables
    total_steps = 0
    log_interval = 1  # Log every batch
    best_loss = float('inf')
    
    def log_metrics(epoch, batch_idx, loss, lr, perplexity, is_val=False, total_batches=None):
        if local_rank == 0:  # Only print from master process
            prefix = "Val" if is_val else "Train"
            progress = f"[{batch_idx}/{total_batches}]" if total_batches else f"[{batch_idx}]"
            print(f"Epoch {epoch+1} {progress} | "
                  f"{prefix} Loss: {loss:.4f} | "
                  f"Perplexity: {perplexity:.2f} | "
                  f"Learning Rate: {lr:.6f}")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = torch.zeros(1).to(device)
        num_batches = 0
        start_time = time.time()
        
        optimizer.zero_grad()
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        for batch_idx, (inputs, targets) in enumerate(train_batches):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            if use_amp:
                with autocast():
                    outputs = model(inputs)
                    outputs = outputs.reshape(-1, outputs.size(-1))
                    targets = targets.reshape(-1)
                    loss = criterion(outputs, targets) / gradient_accumulation_steps
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                    if warmup_steps > 0:
                        scheduler.step()
                        current_lr = scheduler.get_last_lr()[0]
            else:
                outputs = model(inputs)
                outputs = outputs.reshape(-1, outputs.size(-1))
                targets = targets.reshape(-1)
                loss = criterion(outputs, targets) / gradient_accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    if warmup_steps > 0:
                        scheduler.step()
                        current_lr = scheduler.get_last_lr()[0]
            
            # Track loss and metrics
            total_loss += loss.item() * gradient_accumulation_steps
            num_batches += 1
            total_steps += 1
            
            # Calculate perplexity
            batch_perplexity = torch.exp(loss * gradient_accumulation_steps)
            
            # Log progress
            if (batch_idx + 1) % log_interval == 0:
                avg_loss = loss.item() * gradient_accumulation_steps
                log_metrics(epoch, batch_idx + 1, avg_loss, current_lr, 
                          batch_perplexity.item(), False, len(train_batches))
        
        # Synchronize loss across all processes
        if is_distributed:
            dist.all_reduce(total_loss)
            total_loss /= dist.get_world_size()
        
        avg_loss = total_loss.item() / num_batches
        epoch_perplexity = math.exp(avg_loss)
        
        # Evaluate on validation set
        if val_batches is not None and local_rank in [-1, 0]:
            model.eval()
            val_loss = 0
            val_batches_count = 0
            
            with torch.no_grad():
                for val_batch_idx, (inputs, targets) in enumerate(val_batches):
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    
                    if use_amp:
                        with autocast():
                            outputs = model(inputs)
                            outputs = outputs.reshape(-1, outputs.size(-1))
                            targets = targets.reshape(-1)
                            batch_loss = criterion(outputs, targets)
                    else:
                        outputs = model(inputs)
                        outputs = outputs.reshape(-1, outputs.size(-1))
                        targets = targets.reshape(-1)
                        batch_loss = criterion(outputs, targets)
                    
                    val_loss += batch_loss.item()
                    val_batches_count += 1
                    
                    # Log validation metrics
                    if (val_batch_idx + 1) % log_interval == 0:
                        batch_val_perplexity = torch.exp(batch_loss)
                        log_metrics(epoch, val_batch_idx + 1, batch_loss.item(), 
                                  current_lr, batch_val_perplexity.item(), True, len(val_batches))
            
            val_loss /= val_batches_count
            val_perplexity = math.exp(val_loss)
            
            if local_rank == 0:
                print(f"\nEpoch {epoch+1} Summary | "
                      f"Train Loss: {avg_loss:.4f} | "
                      f"Train Perplexity: {epoch_perplexity:.2f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val Perplexity: {val_perplexity:.2f} | "
                      f"Time: {time.time() - start_time:.2f}s")
                
                # Save best model
                if val_loss < best_loss:
                    best_loss = val_loss
                    if local_rank == 0:
                        print(f"New best model! Loss: {val_loss:.4f}, Perplexity: {val_perplexity:.2f}")
                        # Save checkpoint
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.module.state_dict() if is_distributed else model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': val_loss,
                            'perplexity': val_perplexity
                        }, f'models/checkpoint_epoch_{epoch+1}.pt')
        
        elif local_rank == 0:
            print(f"\nEpoch {epoch+1} Summary | "
                  f"Train Loss: {avg_loss:.4f} | "
                  f"Train Perplexity: {epoch_perplexity:.2f} | "
                  f"Time: {time.time() - start_time:.2f}s")
    
    return model, (train_losses, val_losses)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--master_addr', type=str, default='localhost')
    parser.add_argument('--master_port', type=str, default='29500')
    parser.add_argument('--dist_backend', type=str, default='nccl')
    args = parser.parse_args()
    
    # Setup distributed training if needed
    if args.local_rank != -1:
        setup_distributed(args)
    
    try:
        # Data parameters
        data_path = 'data/enwik8'
        data_url = 'https://codeberg.org/pbm/former/raw/branch/master/data/enwik8.gz'

        # Model hyperparameters - adjusted for BPE tokenization
        d_model = 768
        nhead = 12
        num_layers = 16
        dim_feedforward = 3072
        dropout = 0.2
        attention_dropout = 0.15
        activation_dropout = 0.15
        token_dropout = 0.1
        use_checkpoint = True
        stochastic_depth_prob = 0.1

        # Training hyperparameters
        batch_size = 32
        seq_length = 512  # Reduced sequence length since tokens represent larger units
        num_epochs = 100
        learning_rate = 5e-4
        min_lr = 1e-5
        weight_decay = 0.1
        label_smoothing = 0.1
        gradient_accumulation_steps = 8
        use_mixed_precision = True
        warmup_epochs = 2

        # Load data
        print("Loading data...")
        text = load_data(data_path, data_url)
        print(f"Data loaded: {len(text)} characters")

        # Limit the data size for training
        max_chars = 20000000
        if len(text) > max_chars:
            print(f"Limiting data to first {max_chars} characters for training")
            text = text[:max_chars]

        # Load pre-trained tokenizer
        print("Loading pre-trained tokenizer...")
        try:
            tokenizer = Tokenizer.from_file('bpe-enwik8-tokenizer.json')
            vocab_size = tokenizer.get_vocab_size()
            print(f"Loaded tokenizer with vocabulary size: {vocab_size:,}")
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            print("Please train the tokenizer first using train_tokenizer.py")
            return

        # Encode the text
        print("Encoding text with pre-trained tokenizer...")
        encoded = tokenizer.encode(text)
        data = encoded.ids  # Get token IDs from the encoding

        # Split data into training and validation sets (90% / 10%)
        split_idx = int(len(data) * 0.9)
        train_data = data[:split_idx]
        val_data = data[split_idx:]

        # Modify data loading for distributed training
        if args.local_rank != -1:
            train_batches = create_distributed_batches(train_data, batch_size, seq_length, 
                                                     args.local_rank, args.world_size)
            val_batches = create_distributed_batches(val_data, batch_size, seq_length, 
                                                   args.local_rank, args.world_size) if val_data else None
        else:
            train_batches = create_batches(train_data, batch_size, seq_length)
            val_batches = create_batches(val_data, batch_size, seq_length)
        
        # Calculate warmup steps after creating batches
        warmup_steps = len(train_batches) * warmup_epochs

        # Create enhanced model with stochastic depth
        model = EnhancedCharTransformer(
            vocab_size=vocab_size,  # Use BPE vocab size
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            token_dropout=token_dropout,
            use_checkpoint=use_checkpoint,
            stochastic_depth_prob=stochastic_depth_prob
        )

        # Move model to device
        device = torch.device(f"cuda:{args.local_rank}" if args.local_rank != -1 else "cuda")
        model = model.to(device)

        # Print model architecture
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model Parameters: {trainable_params:,} trainable out of {total_params:,} total")

        # Train model with enhanced settings
        print("\n=== Training Enhanced Transformer Model with BPE Tokenization ===")
        model, (train_losses, val_losses) = train_distributed_model(
            model=model,
            train_batches=train_batches,
            val_batches=val_batches,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            min_lr=min_lr,
            device=device,
            patience=8,
            label_smoothing=label_smoothing,
            gradient_accumulation_steps=gradient_accumulation_steps,
            use_mixed_precision=use_mixed_precision,
            use_cosine_schedule=True,
            local_rank=args.local_rank
        )

        # Save model and visualize results only on rank 0
        if args.local_rank in [-1, 0]:
            visualize_results(train_losses, val_losses, 'bpe_transformer_loss.png')
            torch.save({
                'model_state_dict': model.module.state_dict() if args.local_rank != -1 else model.state_dict(),
                'vocab_size': vocab_size,
                'd_model': d_model,
                'nhead': nhead,
                'num_layers': num_layers,
                'dim_feedforward': dim_feedforward,
                'dropout': dropout,
                'attention_dropout': attention_dropout,
                'activation_dropout': activation_dropout,
                'token_dropout': token_dropout,
                'stochastic_depth_prob': stochastic_depth_prob
            }, 'bpe_transformer_model.pt')
    
    finally:
        # Cleanup distributed training
        if args.local_rank != -1:
            cleanup_distributed()

if __name__ == "__main__":
    main()
