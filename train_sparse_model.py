import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import math
from sparse_char_transformer import SparseCharTransformer, generate_square_subsequent_mask
from stable_char_transformer import ByteTokenizer, create_batches, load_data
from contextlib import nullcontext
from torch.cuda.amp import GradScaler, autocast
import gc
import torch.nn.functional as F

def train_sparse_model(model, train_batches, val_batches=None, num_epochs=100,
                      learning_rate=1e-4, weight_decay=0.02, warmup_steps=2000,
                      device='cuda', patience=8, min_lr=1e-5,
                      gradient_accumulation_steps=16, use_mixed_precision=True):
    """Train the sparse transformer model with advanced training techniques"""
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing = True
    
    # Setup optimizer with more stable settings
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-6,  # Start small
        weight_decay=weight_decay,
        eps=1e-8,
        betas=(0.9, 0.95)  # More stable momentum
    )
    
    # Simpler learning rate schedule
    def get_lr(step):
        if step < warmup_steps:
            return learning_rate * (step / warmup_steps)
        return max(min_lr, learning_rate * 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (num_epochs * len(train_batches)))))
    
    # Setup mixed precision training with stable settings
    scaler = GradScaler(
        init_scale=2**10,  # More reasonable initial scale
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=100
    ) if use_mixed_precision else None
    
    # Training metrics
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    global_step = 0
    current_lr = optimizer.param_groups[0]['lr']  # Initialize current_lr
    
    print(f"Initial learning rate: {current_lr:.6f}")
    print(f"Target learning rate: {learning_rate:.6f}")
    print(f"Warmup steps: {warmup_steps}")
    
    # Loss function with less smoothing
    def compute_loss(output, target):
        return F.cross_entropy(
            output.reshape(-1, output.size(-1)),
            target.reshape(-1),
            ignore_index=-1,
            label_smoothing=0.1
        )
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        num_batches = 0
        optimizer.zero_grad(set_to_none=True)  # More efficient gradient clearing
        
        # Training phase
        for batch_idx, batch in enumerate(train_batches):
            try:
                # Update learning rate at the start of each batch
                if batch_idx == 0 or (batch_idx + 1) % gradient_accumulation_steps == 0:
                    current_lr = get_lr(global_step)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = current_lr

                # Handle batch data
                if isinstance(batch, (tuple, list)):
                    batch_data = batch[0]
                else:
                    batch_data = batch
                    
                batch_data = batch_data.to(device)
                    
                # Split into input and target
                input_ids = batch_data[:, :-1]
                target_ids = batch_data[:, 1:]
                
                # Create attention mask for training
                src_mask = generate_square_subsequent_mask(input_ids.size(1)).to(device)
                
                # Forward pass with mixed precision
                with autocast() if use_mixed_precision else nullcontext():
                    output, _ = model(input_ids, src_mask=src_mask)
                    loss = compute_loss(output, target_ids)
                    loss = loss / gradient_accumulation_steps
                
                # Backward pass with mixed precision
                if use_mixed_precision:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Step optimization after accumulation
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    if use_mixed_precision:
                        scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        
                        if torch.isfinite(grad_norm):
                            scaler.step(optimizer)
                            scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        if torch.isfinite(grad_norm):
                            optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                    
                    # Update step counter after optimization
                    global_step += 1
                
                # Add the full (unscaled) loss to total
                total_train_loss += loss.item() * gradient_accumulation_steps
                num_batches += 1
                
                # Progress logging
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx}/{len(train_batches)} | "
                          f"Loss: {loss.item() * gradient_accumulation_steps:.4f} | LR: {current_lr:.6f}")
                    
                # Clear memory
                del output, loss
                if batch_idx % 10 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("WARNING: out of memory, skipping batch")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    optimizer.zero_grad(set_to_none=True)
                    if use_mixed_precision:
                        scaler.update()  # Make sure to update scaler state on OOM
                    continue
                else:
                    raise e
        
        # Calculate average training loss properly
        avg_train_loss = total_train_loss / num_batches if num_batches > 0 else float('inf')
        train_losses.append(avg_train_loss)
        
        # Validation phase
        if val_batches:
            model.eval()
            total_val_loss = 0
            num_val_batches = 0
            
            with torch.no_grad():
                for batch in val_batches:
                    try:
                        # Handle batch data
                        if isinstance(batch, (tuple, list)):
                            batch_data = batch[0]
                        else:
                            batch_data = batch
                            
                        batch_data = batch_data.to(device)
                        
                        # Split into input and target
                        input_ids = batch_data[:, :-1]
                        target_ids = batch_data[:, 1:]
                        src_mask = generate_square_subsequent_mask(input_ids.size(1)).to(device)
                        
                        output, _ = model(input_ids, src_mask=src_mask)
                        loss = compute_loss(output, target_ids)
                        
                        total_val_loss += loss.item()
                        num_val_batches += 1
                        
                        # Clear memory
                        del output, loss
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print("WARNING: out of memory during validation, skipping batch")
                            if hasattr(torch.cuda, 'empty_cache'):
                                torch.cuda.empty_cache()
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
                }, 'models/best_sparse_transformer.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        # Print epoch statistics
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        if val_batches:
            print(f"Average Validation Loss: {avg_val_loss:.4f}")
            print(f"Best Validation Loss: {best_val_loss:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[-1]['lr']:.6f}")
        
        # Clear memory at end of epoch
        gc.collect()
        torch.cuda.empty_cache()
    
    return train_losses, val_losses

def visualize_sparse_attention(model, tokenizer, text, output_dir='attention_analysis'):
    """Visualize attention patterns from the sparse transformer"""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    # Encode text
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
    
    # Get attention weights
    with torch.no_grad():
        _, attention_weights = model(input_ids)
    
    # Plot attention patterns for each layer
    for layer_idx, layer_attention in enumerate(attention_weights):
        if layer_attention is not None:
            weights = layer_attention[0]  # Take first batch
            
            for head_idx in range(weights.size(0)):
                plt.figure(figsize=(10, 10))
                plt.imshow(weights[head_idx].cpu(), cmap='viridis')
                plt.title(f'Layer {layer_idx}, Head {head_idx}')
                plt.xlabel('Key position')
                plt.ylabel('Query position')
                plt.colorbar()
                plt.savefig(f'{output_dir}/attention_layer{layer_idx}_head{head_idx}.png')
                plt.close()

def main():
    # Model configuration with stable settings
    config = {
        'vocab_size': 256,
        'd_model': 512,
        'nhead': 8,
        'num_layers': 12,
        'dim_feedforward': 1024,  # Reduced from 2048
        'dropout': 0.1,
        'activation': 'gelu',
        'use_adaptive_attention': False  # Use standard sparse attention
    }
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model instance
    model = SparseCharTransformer(**config)
    model = model.to(device)
    
    # Create tokenizer
    tokenizer = ByteTokenizer()
    
    # Load training data
    print("Loading training data...")
    train_text = load_data('data/enwik8')
    
    # Split into train/val
    split_idx = int(len(train_text) * 0.9)
    train_data = tokenizer.encode(train_text[:split_idx])
    val_data = tokenizer.encode(train_text[split_idx:])
    
    # Create batches with conservative sizes
    batch_size = 32  # From 16
    seq_length = 384  # If currently using a larger value
    train_batches = create_batches(train_data, batch_size, seq_length)
    val_batches = create_batches(val_data, batch_size, seq_length)
    
    # Train model with stable settings
    print("Training sparse transformer model...")
    train_losses, val_losses = train_sparse_model(
        model=model,
        train_batches=train_batches,
        val_batches=val_batches,
        num_epochs=100,
        learning_rate=1e-4,  # Conservative learning rate
        weight_decay=0.005,  # Reduce weight decay
        warmup_steps=4000,  # Longer warmup
        device=device,
        patience=8,
        gradient_accumulation_steps=8,  # Reduce gradient accumulation
        use_mixed_precision=True  # Keep mixed precision for speed
    )
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'train_losses': train_losses,
        'val_losses': val_losses
    }, 'models/final_sparse_transformer.pt')
    
    # Visualize training history
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Sparse Transformer Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('sparse_transformer_loss.png')
    plt.close()
    
    # Generate and visualize attention patterns
    print("\nGenerating attention visualizations...")
    sample_text = "The quick brown fox jumps over the lazy dog."
    visualize_sparse_attention(model, tokenizer, sample_text)
    
    print("\nTraining complete! Model saved and attention patterns visualized.")

if __name__ == "__main__":
    main() 