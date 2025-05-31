import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import math
import time
import logging
from pathlib import Path
from sparse_transformer_v2 import SparseTransformer
from stable_char_transformer import ByteTokenizer, create_batches, load_data
from torch.cuda.amp import GradScaler, autocast
from contextlib import nullcontext
import gc

# Setup logging
logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('logs/sparse_transformer_training.log'),
        logging.StreamHandler()
    ]
)

def create_sparse_transformer(vocab_size=256):
    """Create sparse transformer model with configuration matching the dense baseline."""
    model = SparseTransformer(
        vocab_size=vocab_size,
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="gelu",
        max_seq_length=1024
    )
    
    # Initialize weights properly
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
    return model

def compute_bpb(loss):
    """Convert loss to bits per byte metric."""
    return float(loss) / math.log(2)  # Ensure float conversion

def compute_metrics(loss, num_tokens):
    """Compute bits per byte and perplexity metrics with proper normalization."""
    bpb = compute_bpb(loss)
    # Properly compute perplexity with numerical stability
    ppl = torch.exp(torch.tensor(min(loss, 20))).item()  # Lower cap for better stability
    return bpb, ppl

def train_epoch(model, train_batches, criterion, optimizer, scheduler, device, use_amp=True):
    model.train()
    total_loss = 0
    total_tokens = 0
    start_time = time.time()
    
    # Create gradient scaler for AMP
    scaler = GradScaler() if use_amp else None
    
    for batch_idx, batch in enumerate(train_batches):
        try:
            # Handle batch data
            if isinstance(batch, (tuple, list)):
                batch_data = batch[0]
            else:
                batch_data = batch
            batch_data = batch_data.to(device)
            
            # Split into input and target
            input_ids = batch_data[:, :-1].contiguous()
            target_ids = batch_data[:, 1:].contiguous()
            
            # Debug input shapes and values
            if batch_idx == 0:
                logging.info(f"Input shape: {input_ids.shape}, Target shape: {target_ids.shape}")
                logging.info(f"Input range: [{input_ids.min().item()}, {input_ids.max().item()}]")
                logging.info(f"Target range: [{target_ids.min().item()}, {target_ids.max().item()}]")
            
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with mixed precision
            with autocast() if use_amp else nullcontext():
                output = model(input_ids)
                output = output.reshape(-1, output.size(-1))
                target_ids = target_ids.reshape(-1)
                
                # Debug output shape and values
                if batch_idx == 0:
                    logging.info(f"Output shape: {output.shape}, Reshaped target shape: {target_ids.shape}")
                    logging.info(f"Output logits range: [{output.min().item()}, {output.max().item()}]")
                    probs = torch.softmax(output[:5], dim=-1)
                    logging.info(f"Sample probabilities: max={probs.max().item()}, min={probs.min().item()}")
                
                loss = criterion(output, target_ids)
                
                # Scale loss by sequence length for better stability
                loss = loss / math.log(2)  # Convert to bits
            
            # Debug loss value
            if batch_idx == 0:
                logging.info(f"Raw loss value: {loss.item()}")
                logging.info(f"Batch {batch_idx} loss stats - Mean: {output.mean().item():.4f}, Std: {output.std().item():.4f}")
            
            # Backward pass with mixed precision
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Reduced from 1.0
                
                # Debug gradient norm
                if batch_idx % 100 == 0:
                    logging.info(f"Gradient norm: {grad_norm.item():.4f}")
                
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            
            # Update metrics - only count non-padding tokens
            batch_tokens = target_ids.numel()
            total_tokens += batch_tokens
            total_loss += loss.item() * batch_tokens
            
            if batch_idx % 100 == 0:
                ms_per_batch = (time.time() - start_time) * 1000 / (batch_idx + 1)
                cur_loss = total_loss / total_tokens
                cur_bpb, cur_ppl = compute_metrics(cur_loss, total_tokens)
                
                # Get current learning rate
                current_lr = optimizer.param_groups[0]['lr']
                
                logging.info(
                    f'Train batch {batch_idx:5d}/{len(train_batches):5d} | '
                    f'ms/batch {ms_per_batch:5.2f} | '
                    f'loss {cur_loss:.4f} | '
                    f'bpb {cur_bpb:5.2f} | '
                    f'ppl {cur_ppl:8.2f} | '
                    f'lr {current_lr:.2e} | '
                    f'grad_norm {grad_norm.item():.2e}'
                )
            
            # Clear memory periodically
            if batch_idx % 10 == 0:
                del output, loss
                gc.collect()
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                logging.warning("WARNING: out of memory, skipping batch")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.update()
                continue
            else:
                raise e
    
    avg_loss = total_loss / total_tokens
    return avg_loss

def evaluate(model, val_batches, criterion, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_batches):
            if isinstance(batch, (tuple, list)):
                batch_data = batch[0]
            else:
                batch_data = batch
            batch_data = batch_data.to(device)
            
            # Split into input and target
            input_ids = batch_data[:, :-1].contiguous()
            target_ids = batch_data[:, 1:].contiguous()
            
            # Forward pass
            output = model(input_ids)
            output = output.reshape(-1, output.size(-1))
            target_ids = target_ids.reshape(-1)
            
            loss = criterion(output, target_ids)
            
            # Update metrics - only count actual target tokens
            batch_tokens = target_ids.numel()
            total_tokens += batch_tokens
            total_loss += loss.item() * batch_tokens
            
            # Debug validation metrics periodically
            if batch_idx % 100 == 0:
                cur_loss = total_loss / total_tokens
                cur_bpb, cur_ppl = compute_metrics(cur_loss, total_tokens)
                logging.info(
                    f'Validation batch {batch_idx:5d}/{len(val_batches):5d} | '
                    f'loss {cur_loss:.4f} | '
                    f'bpb {cur_bpb:5.2f} | '
                    f'ppl {cur_ppl:8.2f}'
                )
    
    avg_loss = total_loss / total_tokens
    return avg_loss

def main():
    # Setup device and clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Create model with gradient checkpointing enabled
    model = create_sparse_transformer()
    model = model.to(device)
    for layer in model.layers:
        layer.use_checkpoint = True
    logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create tokenizer and load data
    tokenizer = ByteTokenizer()
    logging.info("Loading training data...")
    train_text = load_data('data/enwik8')
    
    # Split into train/val with proper ratio
    split_idx = int(len(train_text) * 0.95)  # Increased train ratio
    train_data = tokenizer.encode(train_text[:split_idx])
    val_data = tokenizer.encode(train_text[split_idx:])
    
    # Create batches with adjusted sizes
    batch_size = 32  # Increased from 16
    seq_length = 1024  # Increased from 512
    train_batches = create_batches(train_data, batch_size, seq_length)
    val_batches = create_batches(val_data, batch_size, seq_length)
    
    # Training settings
    num_epochs = 100  # Increased from 50
    warmup_steps = 4000  # Increased from 2000
    base_lr = 1e-3  # Increased from 3e-4
    min_lr = 1e-4  # Increased from 1e-5
    patience = 5  # Increased from 3
    min_delta = 0.005  # Reduced from 0.01 for finer improvements
    
    # Setup training with reduced label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)  # Reduced from 0.1
    optimizer = optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=0.1,  # Increased from 0.01
        betas=(0.9, 0.98)
    )
    
    total_steps = len(train_batches) * num_epochs
    
    # Learning rate schedule with slower decay
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        
        # Slower cosine decay
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress * 0.5))  # Slower decay
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    # Log initial learning rate and schedule parameters
    logging.info(f"Base learning rate: {base_lr:.2e}")
    logging.info(f"Minimum learning rate: {min_lr:.2e}")
    logging.info(f"Initial learning rate: {optimizer.param_groups[0]['lr']:.2e}")
    logging.info(f"Warmup steps: {warmup_steps}")
    logging.info(f"Total steps: {total_steps}")
    
    # Test learning rate schedule
    test_steps = [0, warmup_steps//2, warmup_steps, total_steps//2, total_steps-1]
    for step in test_steps:
        lr = base_lr * lr_lambda(step)
        logging.info(f"Test LR at step {step}: {lr:.2e}")
    
    # Create checkpoint directory
    checkpoint_dir = Path('models/sparse_transformer')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop with improved monitoring
    best_val_bpb = float('inf')
    best_epoch = -1
    metrics_list = []
    patience_counter = 0
    
    logging.info("Starting training...")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Train with AMP enabled
        train_loss = train_epoch(model, train_batches, criterion, optimizer, scheduler, device, use_amp=True)
        val_loss = evaluate(model, val_batches, criterion, device)
        
        # Compute metrics with proper normalization
        train_bpb, train_ppl = compute_metrics(train_loss, len(train_batches) * batch_size * seq_length)
        val_bpb, val_ppl = compute_metrics(val_loss, len(val_batches) * batch_size * seq_length)
        
        epoch_time = time.time() - epoch_start_time
        
        logging.info(
            f'Epoch {epoch:3d} | time: {epoch_time:5.2f}s | '
            f'train loss {train_loss:.4f} | train bpb {train_bpb:5.2f} | train ppl {train_ppl:8.2f} | '
            f'valid loss {val_loss:.4f} | valid bpb {val_bpb:5.2f} | valid ppl {val_ppl:8.2f} | '
            f'lr {optimizer.param_groups[0]["lr"]:.2e}'
        )
        
        # Early stopping check based on validation bpb
        if val_bpb < best_val_bpb - min_delta:
            best_val_bpb = val_bpb
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_bpb': train_bpb,
                'val_bpb': val_bpb,
                'train_ppl': train_ppl,
                'val_ppl': val_ppl,
                'model_config': {
                    'vocab_size': model.embedding.num_embeddings,
                    'd_model': model.d_model,
                    'nhead': model.nhead,
                    'num_layers': model.num_layers,
                    'dim_feedforward': model.layers[0].linear1.out_features,
                    'dropout': model.layers[0].dropout.p,
                    'activation': "gelu",
                    'max_seq_length': model.max_seq_length
                }
            }
            torch.save(checkpoint, checkpoint_dir / 'best_model.pt')
            logging.info(f'Saved new best model with validation bpb: {val_bpb:5.2f} at epoch {epoch}')
        else:
            patience_counter += 1
            logging.info(f'Validation bpb did not improve by {min_delta:.4f}. '
                        f'Best: {best_val_bpb:.4f} at epoch {best_epoch}. '
                        f'Patience: {patience_counter}/{patience}')
            
            if patience_counter >= patience:
                logging.info(f'Early stopping triggered after {epoch + 1} epochs. '
                           f'Best validation bpb: {best_val_bpb:.4f} at epoch {best_epoch}')
                break
        
        # Save metrics and regular checkpoint
        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_bpb': train_bpb,
            'val_bpb': val_bpb,
            'train_ppl': train_ppl,
            'val_ppl': val_ppl,
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        metrics_list.append(metrics)
        
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pt')
        
        # Clear memory at end of epoch
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main() 