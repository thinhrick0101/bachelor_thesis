import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import math
from sparse_char_transformer import SparseCharTransformer, generate_square_subsequent_mask
from stable_char_transformer import ByteTokenizer, create_batches, load_data
from contextlib import nullcontext
import torch.amp as amp

def train_sparse_model(model, train_batches, val_batches=None, num_epochs=100,
                      learning_rate=1e-4, weight_decay=0.1, warmup_steps=1000,
                      device='cuda', patience=8, min_lr=1e-5,
                      gradient_accumulation_steps=4, use_mixed_precision=True):
    """Train the sparse transformer model with advanced training techniques"""
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=warmup_steps, T_mult=2, eta_min=min_lr
    )
    
    # Setup mixed precision training
    scaler = torch.amp.GradScaler('cuda') if use_mixed_precision else None
    
    # Training metrics
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        num_batches = 0
        optimizer.zero_grad()
        
        # Training phase
        for batch_idx, batch in enumerate(train_batches):
            # Handle batch data
            if isinstance(batch, (tuple, list)):
                # If batch is a sequence, take the first element as our data
                batch_data = batch[0]
            else:
                batch_data = batch
                
            # Ensure batch_data is a tensor and on the correct device
            if not isinstance(batch_data, torch.Tensor):
                batch_data = torch.tensor(batch_data)
            batch_data = batch_data.to(device)
                
            # Split into input and target
            input_ids = batch_data[:, :-1]
            target_ids = batch_data[:, 1:]
            
            # Create attention mask for training
            src_mask = generate_square_subsequent_mask(input_ids.size(1)).to(device)
            
            # Forward pass with mixed precision
            with amp.autocast('cuda') if use_mixed_precision else nullcontext():
                output, attention_weights = model(input_ids, src_mask=src_mask)
                loss = nn.functional.cross_entropy(
                    output.view(-1, output.size(-1)),
                    target_ids.view(-1),
                    ignore_index=-1
                )
                
                # Add sparsity regularization based on attention patterns
                sparsity_loss = 0
                for layer_weights in attention_weights:
                    if layer_weights is not None:
                        # Encourage sparsity in later layers
                        layer_idx = attention_weights.index(layer_weights)
                        sparsity_factor = min(1.0, layer_idx / len(attention_weights))
                        sparsity_loss += torch.mean(torch.abs(layer_weights)) * sparsity_factor
                
                loss = loss + 0.01 * sparsity_loss
                loss = loss / gradient_accumulation_steps
            
            # Backward pass with mixed precision
            if use_mixed_precision:
                scaler.scale(loss).backward()
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
            
            total_train_loss += loss.item() * gradient_accumulation_steps
            num_batches += 1
            
            # Update learning rate
            scheduler.step(epoch + batch_idx / len(train_batches))
            
            # Progress logging
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx}/{len(train_batches)} | "
                      f"Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Calculate average training loss
        avg_train_loss = total_train_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        if val_batches:
            model.eval()
            total_val_loss = 0
            num_val_batches = 0
            
            with torch.no_grad():
                for batch in val_batches:
                    # Handle batch data
                    if isinstance(batch, (tuple, list)):
                        # If batch is a sequence, take the first element as our data
                        batch_data = batch[0]
                    else:
                        batch_data = batch
                        
                    # Ensure batch_data is a tensor and on the correct device
                    if not isinstance(batch_data, torch.Tensor):
                        batch_data = torch.tensor(batch_data)
                    batch_data = batch_data.to(device)
                    
                    # Split into input and target
                    input_ids = batch_data[:, :-1]
                    target_ids = batch_data[:, 1:]
                    src_mask = generate_square_subsequent_mask(input_ids.size(1)).to(device)
                    
                    output, _ = model(input_ids, src_mask=src_mask)
                    loss = nn.functional.cross_entropy(
                        output.view(-1, output.size(-1)),
                        target_ids.view(-1),
                        ignore_index=-1
                    )
                    
                    total_val_loss += loss.item()
                    num_val_batches += 1
            
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
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
    
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
    # Model configuration
    config = {
        'vocab_size': 256,
        'd_model': 512,
        'nhead': 8,
        'num_layers': 12,
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'activation': 'relu',
        'use_adaptive_attention': True  # Use adaptive sparse attention
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
    
    # Create batches
    batch_size = 32
    seq_length = 1024
    train_batches = create_batches(train_data, batch_size, seq_length)
    val_batches = create_batches(val_data, batch_size, seq_length)
    
    # Train model
    print("Training sparse transformer model...")
    train_losses, val_losses = train_sparse_model(
        model=model,
        train_batches=train_batches,
        val_batches=val_batches,
        num_epochs=100,
        learning_rate=1e-4,
        weight_decay=0.1,
        warmup_steps=1000,
        device=device,
        patience=8,
        gradient_accumulation_steps=4,
        use_mixed_precision=True
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