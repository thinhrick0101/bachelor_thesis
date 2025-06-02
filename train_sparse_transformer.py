import os
import math
import time
import torch
import torch.nn as nn
import gc
from torch.cuda.amp import GradScaler, autocast
from byte_dataset import create_dataloaders
from sparse_byte_transformer import SparseByteTransformer
import matplotlib.pyplot as plt
from contextlib import nullcontext
from stable_char_transformer import ByteTokenizer


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


def check_gradients(model):
    """Check if all gradients are valid (finite)."""
    valid_gradients = True
    for name, param in model.named_parameters():
        if param.grad is not None:
            if not torch.isfinite(param.grad).all():
                print(f"Invalid gradients detected in {name}")
                valid_gradients = False
    return valid_gradients


def train_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    scaler,
    device,
    epoch,
    grad_clip=0.5  # Reduced from 1.0 for more stability
):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    start_time = time.time()
    num_updates = 0
    
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
                    target.view(-1),
                    label_smoothing=0.1  # Add label smoothing for better generalization
                )
            
            # Check if loss is valid
            if not torch.isfinite(loss):
                print(f"Non-finite loss detected: {loss.item()}")
                optimizer.zero_grad(set_to_none=True)
                continue
                
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Check gradients before optimization
            if not check_gradients(model):
                print("Invalid gradients detected, skipping batch")
                optimizer.zero_grad(set_to_none=True)
                continue
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            # Skip step if gradient norm is not finite
            if not torch.isfinite(grad_norm):
                print(f"Invalid gradient norm detected: {grad_norm}")
                optimizer.zero_grad(set_to_none=True)
                continue
            
            # Optimizer step with scaling
            scaler.step(optimizer)
            scale = scaler.get_scale()
            scaler.update()
            
            # Check if optimizer step was skipped due to gradient scaling
            if scale <= scaler.get_scale():
                num_updates += 1
                if scheduler is not None:
                    scheduler.step()
            
            optimizer.zero_grad(set_to_none=True)
            
            # Logging
            total_loss += loss.item()
            if batch_idx % 50 == 0:
                ms_per_batch = (time.time() - start_time) * 1000 / (batch_idx + 1)
                cur_loss = total_loss / (batch_idx + 1)
                cur_bpb = calculate_bpb(cur_loss)
                lr = optimizer.param_groups[0]['lr']
                print(f'| epoch {epoch:3d} | {batch_idx:5d}/{len(train_loader):5d} batches | '
                      f'ms/batch {ms_per_batch:5.2f} | loss {cur_loss:5.2f} | bpb {cur_bpb:5.2f} | '
                      f'lr {lr:.2e} | updates {num_updates}')
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


def visualize_loss(train_losses, val_losses=None, output_file='sparse_model_loss.png'):
    """Visualize training and validation losses"""
    plt.figure(figsize=(12, 6))
    
    # Plot training loss
    plt.plot(train_losses, label='Training Loss', marker='o', markersize=4, linestyle='-', linewidth=1)
    
    # Plot validation loss if available
    if val_losses:
        plt.plot(val_losses, label='Validation Loss', marker='s', markersize=4, linestyle='-', linewidth=1)
        
        # Plot best validation loss point
        best_epoch = val_losses.index(min(val_losses))
        best_loss = val_losses[best_epoch]
        plt.plot(best_epoch, best_loss, 'r*', markersize=10, label=f'Best Val Loss: {best_loss:.4f}')
    
    plt.title('Sparse Transformer Training History', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add bits-per-byte as secondary y-axis
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # Create BPB ticks based on loss values
    loss_ticks = ax1.get_yticks()
    bpb_ticks = [x / math.log(2) for x in loss_ticks if x > 0]
    ax2.set_yticks(bpb_ticks)
    ax2.set_yticklabels([f'{x:.2f}' for x in bpb_ticks])
    ax2.set_ylabel('Bits per Byte', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print statistics
    print("\nTraining Statistics:")
    print(f"Initial Loss: {train_losses[0]:.4f}")
    print(f"Final Loss: {train_losses[-1]:.4f}")
    print(f"Best Loss: {min(train_losses):.4f}")
    
    if val_losses:
        print("\nValidation Statistics:")
        print(f"Initial Loss: {val_losses[0]:.4f}")
        print(f"Final Loss: {val_losses[-1]:.4f}")
        print(f"Best Loss: {min(val_losses):.4f}")


def generate_text(model, tokenizer, prompt, max_length=1000, temperature=0.7, top_k=50, top_p=0.9, device='cuda'):
    """Generate text using the trained sparse transformer with improved sampling"""
    model.eval()
    
    # Encode the prompt
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    generated = input_tensor
    
    # Track generated tokens for repetition penalty
    generated_tokens = set()
    
    # Generate text token by token
    with torch.no_grad():
        for _ in range(max_length):
            # Get model predictions
            with autocast():
                logits = model(generated)
                next_token_logits = logits[0, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply repetition penalty
            for token in generated_tokens:
                next_token_logits[token] /= 1.2  # Penalize repeated tokens
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token with temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Add to generated tokens set
            generated_tokens.add(next_token.item())
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            
            # Stop if we generate a newline or end token
            if next_token.item() in [10, 0]:  # newline or end token
                break
            
            # Stop if we detect repetitive pattern
            if len(generated) > 10:
                last_tokens = generated[0, -10:].tolist()
                if len(set(last_tokens)) <= 2:  # If using only 1-2 tokens repeatedly
                    break
    
    # Decode and return the generated text
    return tokenizer.decode(generated[0].tolist())


def train_model(model, train_batches, val_batches=None, num_epochs=30,
                learning_rate=1e-4, weight_decay=0.1, warmup_steps=4000,
                device='cuda', patience=5, min_lr=1e-5,
                gradient_accumulation_steps=8, use_mixed_precision=True):
    """Train the sparse transformer model with advanced training techniques"""
    
    # Setup optimizer with stronger regularization
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate / 10,  # Start at lower learning rate
        weight_decay=weight_decay,  # Increased weight decay
        betas=(0.9, 0.98),     # Standard transformer betas
        eps=1e-8
    )
    
    # Learning rate scheduler with faster warmup
    def get_lr(step):
        if step < warmup_steps:
            return learning_rate * min((step + 1) / warmup_steps, 
                                     ((step + 1) / warmup_steps) ** 2)
        progress = (step - warmup_steps) / (num_epochs * len(train_batches))
        return max(min_lr, learning_rate * 0.5 * (1 + math.cos(math.pi * progress)))
    
    # Loss function with token-level entropy regularization
    def compute_loss(output, target, reduction='mean'):
        # Standard cross entropy
        ce_loss = nn.functional.cross_entropy(
            output.view(-1, 256),
            target.view(-1),
            reduction=reduction
        )
        
        # Add entropy regularization to prevent mode collapse
        probs = torch.softmax(output.view(-1, 256), dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
        
        # Combine losses with entropy encouragement
        return ce_loss - 0.1 * entropy  # Encourage diversity
    
    # Setup mixed precision training
    scaler = GradScaler(
        init_scale=2**10,      # Higher initial scale
        growth_factor=2,       # Faster growth
        backoff_factor=0.5,
        growth_interval=2000,
        enabled=use_mixed_precision
    ) if use_mixed_precision else None
    
    # Training metrics
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    global_step = 0
    
    # Initialize learning rate
    current_lr = get_lr(0)
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        num_batches = 0
        start_time = time.time()
        
        # Training phase
        for batch_idx, (data, target) in enumerate(train_batches):
            try:
                # Update learning rate
                if batch_idx == 0 or (batch_idx + 1) % gradient_accumulation_steps == 0:
                    current_lr = get_lr(global_step)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = current_lr
                
                # Move data to device
                data = data.to(device)
                target = target.to(device)
                
                # Forward pass with mixed precision
                with autocast(enabled=use_mixed_precision):
                    output = model(data)
                    loss = compute_loss(output, target) / gradient_accumulation_steps
                
                # Check if loss is valid
                if not torch.isfinite(loss):
                    print(f"Warning: Non-finite loss detected: {loss.item()}")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                
                # Backward pass with gradient scaling
                if use_mixed_precision:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Gradient accumulation step
                if (batch_idx + 1) % gradient_accumulation_steps == 0 or batch_idx == len(train_batches) - 1:
                    # Unscale gradients for clipping
                    if use_mixed_precision:
                        scaler.unscale_(optimizer)
                    
                    # Gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=1.0,
                        error_if_nonfinite=False
                    )
                    
                    # Skip step if gradient norm is not finite
                    if not torch.isfinite(grad_norm):
                        print(f"Warning: Invalid gradient norm detected: {grad_norm}")
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    
                    # Optimizer step
                    if use_mixed_precision:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                
                # Update metrics
                total_train_loss += loss.item() * gradient_accumulation_steps
                num_batches += 1
                
                # Progress logging
                if batch_idx % 100 == 0:
                    ms_per_batch = (time.time() - start_time) * 1000 / (batch_idx + 1)
                    cur_loss = total_train_loss / num_batches
                    cur_bpb = calculate_bpb(cur_loss)
                    print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx}/{len(train_batches)} | "
                          f"Loss: {cur_loss:.4f} | BPB: {cur_bpb:.4f} | "
                          f"LR: {current_lr:.6f} | ms/batch: {ms_per_batch:.1f}")
                
                # Memory management
                if batch_idx % 500 == 0:
                    clear_gpu_memory()
            
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("Warning: OOM, clearing cache and skipping batch")
                    clear_gpu_memory()
                    if 'data' in locals(): del data
                    if 'target' in locals(): del target
                    if 'output' in locals(): del output
                    if 'loss' in locals(): del loss
                    continue
                else:
                    raise e
        
        # Calculate average training loss
        avg_train_loss = total_train_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        if val_batches:
            model.eval()
            total_val_loss = 0
            num_val_batches = 0
            
            with torch.no_grad():
                for data, target in val_batches:
                    try:
                        data = data.to(device)
                        target = target.to(device)
                        
                        with autocast(enabled=use_mixed_precision):
                            output = model(data)
                            # Use same loss function as training
                            loss = compute_loss(output, target)
                        
                        total_val_loss += loss.item()
                        num_val_batches += 1
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print("Warning: OOM during validation, skipping batch")
                            clear_gpu_memory()
                            continue
                        else:
                            raise e
            
            avg_val_loss = total_val_loss / num_val_batches
            val_losses.append(avg_val_loss)
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'config': model.config if hasattr(model, 'config') else None
                }, 'models/best_sparse_transformer.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"Training Loss: {avg_train_loss:.4f} (BPB: {calculate_bpb(avg_train_loss):.4f})")
        if val_batches:
            print(f"Validation Loss: {avg_val_loss:.4f} (BPB: {calculate_bpb(avg_val_loss):.4f})")
            print(f"Best Validation Loss: {best_val_loss:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Clear memory at end of epoch
        clear_gpu_memory()
    
    return model, (train_losses, val_losses)


def main():
    # Model configuration with more stable settings
    config = {
        'd_model': 384,          # Reduced from 512
        'nhead': 6,             # Reduced from 8
        'num_layers': 8,        # Reduced from 12
        'dim_feedforward': 1536, # Reduced from 2048
        'dropout': 0.2,         # Increased from 0.1
        'attention_dropout': 0.2, # Increased from 0.1
        'token_dropout': 0.1,    # Added token dropout
        'max_len': 512          # Reduced from 1024
    }
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model instance
    model = SparseByteTransformer(**config)
    model = model.to(device)
    
    # Create tokenizer
    tokenizer = ByteTokenizer()
    
    # Check if model exists
    model_path = 'models/sparse_byte_transformer.pt'
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Create dataloaders with smaller batch size
        train_loader, val_loader = create_dataloaders(
            train_path=os.path.join("data", "enwik8_splits", "train.bin"),
            val_path=os.path.join("data", "enwik8_splits", "val.bin"),
            seq_length=config['max_len'],
            batch_size=8,        # Reduced from 16
            num_workers=2        # Reduced from 4
        )
        
        # Train model with more stable settings
        print("Training model...")
        model, (train_losses, val_losses) = train_model(
            model=model,
            train_batches=train_loader,
            val_batches=val_loader,
            num_epochs=1,
            learning_rate=1e-4,  # Reduced from 5e-5
            weight_decay=0.1,   # Reduced from 0.1
            warmup_steps=4000,   # Increased from 4000
            device=device,
            patience=5,          # Reduced from 8
            min_lr=1e-5,        # Reduced from 1e-5
            gradient_accumulation_steps=8,  # Increased from 8
            use_mixed_precision=True
        )
        
        # Save model and loss history
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        print(f"Saving model to {model_path}")
        torch.save({
            'model_state_dict': model.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'config': config
        }, model_path)
        
        # Visualize training history
        print("Generating loss plot...")
        visualize_loss(train_losses, val_losses, 'sparse_model_training_loss.png')
    
    # Generate some example text
    print("\nGenerating example texts with different temperatures:")
    prompt = "The movie was"
    
    print("\nConservative sampling (temperature=0.7):")  # Increased from 0.6
    generated = generate_text(model, tokenizer, prompt, temperature=0.7, max_length=200)
    print(generated)
    
    print("\nBalanced sampling (temperature=0.9):")  # Increased from 0.8
    generated = generate_text(model, tokenizer, prompt, temperature=0.9, max_length=200)
    print(generated)
    
    print("\nCreative sampling (temperature=1.2):")  # Increased from 1.0
    generated = generate_text(model, tokenizer, prompt, temperature=1.2, max_length=200)
    print(generated)
    
    # Interactive generation
    print("\nEnter prompts for text generation (type 'exit' to quit):")
    while True:
        prompt = input("\nPrompt: ")
        if prompt.lower() == 'exit':
            break
            
        temp = float(input("Temperature (0.1-1.2): "))  # Increased max temp
        length = int(input("Maximum length: "))
        
        generated = generate_text(
            model, 
            tokenizer, 
            prompt, 
            temperature=temp,
            max_length=length
        )
        print("\nGenerated text:")
        print(generated)


if __name__ == "__main__":
    main() 