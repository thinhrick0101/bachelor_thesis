"""
Training script for the SparseByteTransformer model on Enwik8 dataset.
"""

import os
import math
import time
import torch
import torch.nn as nn
import gc
from torch.cuda.amp import GradScaler, autocast
from byte_dataset import create_dataloaders
from sparse_byte_transformer import SparseByteTransformer


def clear_gpu_memory():
    """Clear GPU memory cache and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def print_gpu_memory():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_reserved = torch.cuda.max_memory_reserved() / 1024**3
        print(f"\nGPU Memory: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Peak={max_reserved:.2f}GB")


def calculate_bpb(loss):
    """Convert cross entropy loss to bits per byte (BPB)."""
    return loss / math.log(2)


def train_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    scaler,
    device,
    epoch,
    grad_clip=1.0
):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        try:
            # Move to device
            data = data.to(device)
            target = target.to(device)
            
            # Forward pass with mixed precision
            with autocast():
                output = model(data)
                loss = nn.functional.cross_entropy(
                    output.view(-1, 256),
                    target.view(-1)
                )
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            # Optimizer step with scaling
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            if scheduler is not None:
                scheduler.step()
            
            # Logging
            total_loss += loss.item()
            if batch_idx % 50 == 0:
                ms_per_batch = (time.time() - start_time) * 1000 / (batch_idx + 1)
                cur_loss = total_loss / (batch_idx + 1)
                cur_bpb = calculate_bpb(cur_loss)
                lr = optimizer.param_groups[0]['lr']
                print(f'| epoch {epoch:3d} | {batch_idx:5d}/{len(train_loader):5d} batches | '
                      f'ms/batch {ms_per_batch:5.2f} | loss {cur_loss:5.2f} | bpb {cur_bpb:5.2f} | lr {lr:.2e}')
                print_gpu_memory()
            
            # Clear memory every 500 batches
            if batch_idx % 500 == 0:
                clear_gpu_memory()
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print('| WARNING: out of memory, clearing cache and skipping batch')
                clear_gpu_memory()
                if 'data' in locals():
                    del data
                if 'target' in locals():
                    del target
                if 'output' in locals():
                    del output
                if 'loss' in locals():
                    del loss
                continue
            else:
                raise e
    
    return total_loss / len(train_loader)


@torch.no_grad()
def evaluate(model, val_loader, device):
    """Evaluate the model on the validation set."""
    model.eval()
    total_loss = 0
    
    for data, target in val_loader:
        try:
            # Move to device
            data = data.to(device)
            target = target.to(device)
            
            # Forward pass
            with autocast():
                output = model(data)
                loss = nn.functional.cross_entropy(
                    output.view(-1, 256),
                    target.view(-1)
                )
            
            total_loss += loss.item()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print('| WARNING: out of memory during evaluation, clearing cache and skipping batch')
                clear_gpu_memory()
                continue
            else:
                raise e
    
    avg_loss = total_loss / len(val_loader)
    avg_bpb = calculate_bpb(avg_loss)
    
    return avg_loss, avg_bpb


def main():
    # Clear memory before starting
    clear_gpu_memory()
    
    # Model configuration
    config = {
        # Smaller model configuration to reduce memory usage
        "model_dim": 384,          # Reduced from 512
        "num_heads": 6,           # Reduced from 8
        "num_layers": 8,          # Reduced from 12
        "ffn_dim": 1536,         # Reduced from 2048
        "dropout": 0.1,
        "attention_dropout": 0.1,
        "token_dropout": 0.0,
        "batch_size": 4,          # Further reduced from 8
        "seq_length": 1024,       # Further reduced from 2048
        "learning_rate": 1e-4,
        "warmup_steps": 4000,
        "grad_clip": 1.0
    }
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        # Print GPU info
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"Available GPU memory: {torch.cuda.mem_get_info()[0] / 1024**3:.1f} GB")
    
    # Create model
    model = SparseByteTransformer(
        d_model=config["model_dim"],
        nhead=config["num_heads"],
        num_layers=config["num_layers"],
        dim_feedforward=config["ffn_dim"],
        dropout=config["dropout"],
        attention_dropout=config["attention_dropout"],
        token_dropout=config["token_dropout"],
        max_len=config["seq_length"]
    ).to(device)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_path=os.path.join("data", "enwik8_splits", "train.bin"),
        val_path=os.path.join("data", "enwik8_splits", "val.bin"),
        seq_length=config["seq_length"],
        batch_size=config["batch_size"],
        num_workers=4
    )
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join("bachelor_thesis", "models", "sparse_transformer")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=0.01
    )
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min((step + 1) / config["warmup_steps"], 1.0)
    )
    
    # Gradient scaler for mixed precision
    scaler = GradScaler()
    
    # Training loop
    best_val_loss = float('inf')
    print("\nStarting training...")
    print('-' * 89)
    
    try:
        for epoch in range(1, 51):  # 50 epochs
            epoch_start_time = time.time()
            
            # Train
            train_loss = train_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                epoch=epoch,
                grad_clip=config["grad_clip"]
            )
            
            # Clear memory before evaluation
            clear_gpu_memory()
            
            # Evaluate
            val_loss, val_bpb = evaluate(model, val_loader, device)
            
            # Print metrics
            print('-' * 89)
            print(f'| end of epoch {epoch:3d} | time: {time.time() - epoch_start_time:5.2f}s | '
                  f'valid loss {val_loss:5.2f} | valid bpb {val_bpb:5.2f}')
            print('-' * 89)
            
            # Save checkpoint if best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                    'val_bpb': val_bpb,
                    'config': config
                }
                checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
                torch.save(checkpoint, checkpoint_path)
                print(f'| saved checkpoint with val_loss {val_loss:5.2f}')
            
            # Clear memory after each epoch
            clear_gpu_memory()
            
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')


if __name__ == '__main__':
    main() 