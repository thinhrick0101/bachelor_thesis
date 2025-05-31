"""
Training script for SparseByteTransformer on the Enwik8 dataset.
This implements byte-level language modeling with sparse attention patterns.
"""

import os
import time
import math
import wandb
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from thop import profile
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from sparse_byte_transformer import SparseByteTransformer
from byte_dataset import ByteDataset, ByteTokenizer


# ==============================
# Metrics and Utilities
# ==============================

def calculate_bpb(loss):
    """Convert cross entropy loss to bits per byte (BPB)."""
    return loss / math.log(2)

def count_parameters(model):
    """Count number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_flops(model, seq_length):
    """
    Calculate FLOPs for one forward pass.
    Args:
        model: The transformer model
        seq_length: Length of input sequence
    Returns:
        flops: Number of floating point operations
    """
    input_tensor = torch.randint(0, 256, (1, seq_length))
    flops, _ = profile(model, inputs=(input_tensor,), verbose=False)
    return flops


# ==============================
# Training Functions
# ==============================

def train_epoch(model, train_loader, optimizer, scheduler, scaler, device, epoch):
    """
    Train for one epoch.
    Args:
        model: The transformer model
        train_loader: DataLoader for training data
        optimizer: The optimizer
        scheduler: Learning rate scheduler
        scaler: Gradient scaler for mixed precision
        device: Device to train on
        epoch: Current epoch number
    Returns:
        average_loss: Average loss over the epoch
    """
    model.train()
    total_loss = 0
    total_tokens = 0
    start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data to device
        data = data.to(device)
        target = target.to(device)
        
        # Forward pass with mixed precision
        with autocast():
            logits = model(data)
            # Flatten predictions and targets for loss calculation
            flat_logits = logits[:, :-1].reshape(-1, 256)
            flat_targets = target[:, 1:].reshape(-1)
            loss = nn.functional.cross_entropy(flat_logits, flat_targets)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        if scheduler is not None:
            scheduler.step()
        
        # Update metrics
        total_loss += loss.item() * data.size(1)
        total_tokens += data.size(1)
        
        # Log progress
        if batch_idx % 100 == 0:
            ms_per_batch = (time.time() - start_time) * 1000 / (batch_idx + 1)
            cur_bpb = calculate_bpb(loss.item())
            print(f'| epoch {epoch:3d} | {batch_idx:5d}/{len(train_loader):5d} batches '
                  f'| ms/batch {ms_per_batch:5.2f} | bpb {cur_bpb:5.2f} |')
            
            # Log to wandb
            wandb.log({
                'train_bpb': cur_bpb,
                'learning_rate': scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr'],
                'ms_per_batch': ms_per_batch
            })
    
    return total_loss / total_tokens

def evaluate(model, val_loader, device):
    """
    Evaluate the model on validation data.
    Args:
        model: The transformer model
        val_loader: DataLoader for validation data
        device: Device to evaluate on
    Returns:
        average_loss: Average loss over the validation set
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            target = target.to(device)
            
            with autocast():
                logits = model(data)
                flat_logits = logits[:, :-1].reshape(-1, 256)
                flat_targets = target[:, 1:].reshape(-1)
                loss = nn.functional.cross_entropy(flat_logits, flat_targets)
            
            total_loss += loss.item() * data.size(1)
            total_tokens += data.size(1)
    
    return total_loss / total_tokens

def generate_sample(model, tokenizer, prefix="The ", max_new_tokens=1000, temperature=0.8):
    """
    Generate text using the trained model.
    Args:
        model: The transformer model
        tokenizer: ByteTokenizer instance
        prefix: Starting text
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature (lower = more focused)
    Returns:
        generated_text: The generated text string
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Convert prefix to tensor
    prefix_tokens = torch.tensor([tokenizer.encode(prefix)], device=device)
    
    # Generate
    with torch.no_grad():
        output_tokens = model.generate(prefix_tokens, max_new_tokens, temperature)
        
    # Decode
    generated_text = tokenizer.decode(output_tokens[0].cpu().numpy())
    return generated_text


# ==============================
# Main Training Loop
# ==============================

def main():
    """Main training function."""
    
    # Initialize wandb
    wandb.init(project="sparse-transformer-enwik8", name="sparse-transformer-run")
    
    # Model and training configuration
    config = {
        # Model architecture
        'd_model': 512,
        'nhead': 8,
        'num_layers': 12,
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'attention_dropout': 0.1,
        'token_dropout': 0.0,
        
        # Training parameters
        'batch_size': 32,
        'seq_length': 1024,
        'learning_rate': 1e-4,
        'warmup_steps': 4000,
        'max_epochs': 100,
        'weight_decay': 0.01
    }
    wandb.config.update(config)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('models/sparse_enwik8', exist_ok=True)
    
    # Initialize model
    model = SparseByteTransformer(
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        attention_dropout=config['attention_dropout'],
        token_dropout=config['token_dropout']
    ).to(device)
    
    # Log model statistics
    n_params = count_parameters(model)
    flops = calculate_flops(model, config['seq_length'])
    print(f'Number of parameters: {n_params:,}')
    print(f'FLOPs per forward pass: {flops:,}')
    wandb.log({'n_params': n_params, 'flops': flops})
    
    # Data loading
    tokenizer = ByteTokenizer()
    train_dataset = ByteDataset(
        path='data/enwik8/train.txt',
        seq_length=config['seq_length']
    )
    val_dataset = ByteDataset(
        path='data/enwik8/val.txt',
        seq_length=config['seq_length']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Optimization
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    def lr_lambda(step):
        if step < config['warmup_steps']:
            return step / config['warmup_steps']
        return 1.0
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(config['max_epochs']):
        epoch_start_time = time.time()
        
        # Train and evaluate
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, device, epoch)
        train_bpb = calculate_bpb(train_loss)
        val_loss = evaluate(model, val_loader, device)
        val_bpb = calculate_bpb(val_loss)
        
        # Generate sample text
        sample_text = generate_sample(model, tokenizer)
        
        # Log metrics
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_bpb': train_bpb,
            'val_loss': val_loss,
            'val_bpb': val_bpb,
            'generated_text': wandb.Html(sample_text)
        })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, 'models/sparse_enwik8/best_model.pt')
        
        # Print progress
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {time.time() - epoch_start_time:5.2f}s | '
              f'valid bpb {val_bpb:8.3f} |')
        print('-' * 89)
        
        # Periodic checkpoints
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, f'models/sparse_enwik8/checkpoint_epoch{epoch}.pt')
    
    wandb.finish()


if __name__ == '__main__':
    main() 