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
import argparse
import signal
import sys
from torch.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint

# Import the original transformer model file
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from stable_char_transformer import CharacterTokenizer, ImprovedPositionalEncoding, EnhancedTransformerBlock, EnhancedCharTransformer, load_data, create_batches, get_linear_schedule_with_warmup, visualize_results

# Import checkpoint utilities
from checkpoint_utils import save_checkpoint, load_checkpoint, find_latest_checkpoint

# Global flag for graceful interruption
INTERRUPT_FLAG = False

# Define handler for SIGTERM
def handle_sigterm(signum, frame):
    global INTERRUPT_FLAG
    print("\nReceived SIGTERM signal. Will save checkpoint and exit after current epoch.")
    INTERRUPT_FLAG = True

# Register the signal handler
signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigterm)

def train_model_with_checkpointing(model, train_batches, val_batches=None, 
                                   num_epochs=5, learning_rate=0.0001,
                                   weight_decay=0.01, warmup_steps=0, 
                                   device=None, patience=3, label_smoothing=0.0,
                                   gradient_accumulation_steps=1, use_mixed_precision=True,
                                   checkpoint_dir='checkpoints', checkpoint_freq=1,
                                   resume_from=None):
    """
    Train the model with checkpointing, resumption, and graceful termination handling
    """
    global INTERRUPT_FLAG
    
    # Determine device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model to device
    model = model.to(device)
    print(f"Training on device: {device}")

    # Initialize mixed precision training if available and requested
    use_amp = use_mixed_precision and device.type == 'cuda'
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print("Using mixed precision training (FP16)")

    # Print gradient accumulation info
    if gradient_accumulation_steps > 1:
        print(f"Using gradient accumulation with {gradient_accumulation_steps} steps")
        effective_batch_size = train_batches[0][0].size(0) * gradient_accumulation_steps
        print(f"Effective batch size: {effective_batch_size}")

    # Define loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

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
    total_steps = len(train_batches) * num_epochs // gradient_accumulation_steps

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
    start_epoch = 0

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Try to load checkpoint if resume_from is specified or if there's a latest checkpoint
    if resume_from is not None or os.path.exists(checkpoint_dir):
        checkpoint_path = resume_from if resume_from else find_latest_checkpoint(checkpoint_dir)
        if checkpoint_path and os.path.exists(checkpoint_path):
            model, optimizer, scheduler, start_epoch, train_losses, val_losses, best_val_loss = load_checkpoint(
                model, optimizer, scheduler, checkpoint_path, device=device
            )
            start_epoch += 1  # Start from the next epoch
            if best_val_loss != float('inf'):
                best_model_state = model.state_dict().copy()
            
            print(f"Resuming training from epoch {start_epoch}")
            print(f"Best validation loss so far: {best_val_loss:.4f}")

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        if INTERRUPT_FLAG:
            print("Interrupted. Saving checkpoint before exiting...")
            save_checkpoint(
                model, optimizer, scheduler, epoch, 
                train_losses, val_losses, best_val_loss,
                checkpoint_dir, f"interrupt_epoch_{epoch}.pt"
            )
            break
            
        model.train()
        total_loss = 0
        num_batches = 0
        start_time = time.time()

        # Zero the gradients at the beginning of each epoch
        optimizer.zero_grad()

        # Process each batch
        for batch_idx, (inputs, targets) in enumerate(train_batches):
            # Move tensors to device
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass with mixed precision if enabled
            if use_amp:
                with autocast(device_type='cuda'):
                    # Forward pass
                    outputs = model(inputs)

                    # Reshape for loss calculation
                    outputs = outputs.reshape(-1, outputs.size(-1))
                    targets = targets.reshape(-1)

                    # Calculate loss
                    loss = criterion(outputs, targets)
                    # Scale loss for gradient accumulation
                    loss = loss / gradient_accumulation_steps

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()

                # Update weights if we've accumulated enough gradients
                if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_batches):
                    # Unscale gradients for clipping
                    scaler.unscale_(optimizer)

                    # Clip gradients to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    # Update weights with gradient scaling
                    scaler.step(optimizer)
                    scaler.update()

                    # Zero gradients after update
                    optimizer.zero_grad()

                    # Update learning rate with warmup scheduler
                    if warmup_steps > 0:
                        scheduler.step()
            else:
                # Standard precision training
                # Forward pass
                outputs = model(inputs)

                # Reshape for loss calculation
                outputs = outputs.reshape(-1, outputs.size(-1))
                targets = targets.reshape(-1)

                # Calculate loss
                loss = criterion(outputs, targets)
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps

                # Backward pass
                loss.backward()

                # Update weights if we've accumulated enough gradients
                if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_batches):
                    # Clip gradients to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    # Update weights
                    optimizer.step()

                    # Zero gradients after update
                    optimizer.zero_grad()

                    # Update learning rate with warmup scheduler
                    if warmup_steps > 0:
                        scheduler.step()

            # Track loss (use the unscaled loss for logging)
            total_loss += loss.item() * gradient_accumulation_steps
            num_batches += 1

            # Print batch progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_batches)}, "
                      f"Loss: {loss.item() * gradient_accumulation_steps:.4f}")
                
            # Check for interruption
            if INTERRUPT_FLAG:
                break

        # Calculate average loss for the epoch
        avg_loss = total_loss / num_batches
        train_losses.append(avg_loss)

        # Calculate perplexity
        perplexity = math.exp(avg_loss)

        # Evaluate on validation set if provided
        if val_batches is not None and not INTERRUPT_FLAG:
            model.eval()
            val_total_loss = 0
            val_num_batches = 0

            with torch.no_grad():
                for inputs, targets in val_batches:
                    # Move tensors to device
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    # Forward pass (use autocast for consistency if mixed precision is enabled)
                    if use_amp:
                        with autocast(device_type='cuda'):
                            outputs = model(inputs)

                            # Reshape for loss calculation
                            outputs = outputs.reshape(-1, outputs.size(-1))
                            targets = targets.reshape(-1)

                            # Calculate loss
                            loss = criterion(outputs, targets)
                    else:
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
                print(f"New best model with validation loss: {best_val_loss:.4f}, perplexity: {math.exp(best_val_loss):.2f}")
                
                # Save best model checkpoint separately
                save_checkpoint(
                    model, optimizer, scheduler, epoch, 
                    train_losses, val_losses, best_val_loss,
                    checkpoint_dir, "best_model.pt"
                )
            else:
                no_improvement_count += 1

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

        # Save checkpoint based on frequency
        if (epoch + 1) % checkpoint_freq == 0 or epoch == num_epochs - 1 or INTERRUPT_FLAG:
            save_checkpoint(
                model, optimizer, scheduler, epoch, 
                train_losses, val_losses, best_val_loss,
                checkpoint_dir
            )
            
        # Update visualization after each epoch
        try:
            visualize_results(train_losses, val_losses, 'training_progress.png')
        except Exception as e:
            print(f"Error creating visualization: {e}")

        # Early stopping
        if no_improvement_count >= patience:
            print(f"No improvement for {patience} epochs. Early stopping.")
            break
            
        # Check for interruption
        if INTERRUPT_FLAG:
            break

    # Load best model if validation was used
    if val_batches is not None and best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation loss: {best_val_loss:.4f}, perplexity: {math.exp(best_val_loss):.2f}")

    return model, (train_losses, val_losses)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train a character-level transformer model')
    parser.add_argument('--data_path', type=str, default='data/enwik8', help='Path to data file')
    parser.add_argument('--data_url', type=str, default='https://codeberg.org/pbm/former/raw/branch/master/data/enwik8.gz', help='URL to download data from')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--resume_from', type=str, default=None, help='Resume training from checkpoint file')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--seq_length', type=int, default=512, help='Sequence length')
    parser.add_argument('--num_epochs', type=int, default=25, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--checkpoint_freq', type=int, default=1, help='Save checkpoint every N epochs')
    parser.add_argument('--max_chars', type=int, default=3000000, help='Maximum number of characters to use for training')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Model hyperparameters - enhanced configuration
    d_model = 512  # Embedding dimension
    nhead = 16  # Number of attention heads
    num_layers = 12  # Number of transformer layers
    dim_feedforward = 2048  # Dimension of feedforward network
    dropout = 0.2  # Dropout rate
    attention_dropout = 0.1  # Attention dropout rate
    activation_dropout = 0.1  # Activation dropout rate
    token_dropout = 0.05  # Token-level dropout rate
    use_checkpoint = True  # Use gradient checkpointing

    # Training hyperparameters
    weight_decay = 0.1  # Weight decay for regularization
    label_smoothing = 0.1  # Label smoothing factor
    gradient_accumulation_steps = 4  # Gradient accumulation steps
    use_mixed_precision = True  # Use mixed precision training
    patience = 5  # Patience for early stopping

    # Load data
    print("Loading data...")
    text = load_data(args.data_path, args.data_url)
    print(f"Data loaded: {len(text)} characters")

    # Limit the data size for training
    if len(text) > args.max_chars:
        print(f"Limiting data to first {args.max_chars} characters for training")
        text = text[:args.max_chars]

    # Create tokenizer
    tokenizer = CharacterTokenizer(text)

    # Encode the text
    data = tokenizer.encode(text)

    # Split data into training and validation sets (90% / 10%)
    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create batches
    print("Creating batches...")
    train_batches = create_batches(train_data, args.batch_size, args.seq_length)
    val_batches = create_batches(val_data, args.batch_size, args.seq_length)
    print(f"Created {len(train_batches)} training batches and {len(val_batches)} validation batches")

    # Create enhanced model
    model = EnhancedCharTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        attention_dropout=attention_dropout,
        activation_dropout=activation_dropout,
        token_dropout=token_dropout,
        use_checkpoint=use_checkpoint
    )

    # Print model architecture
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {trainable_params:,} trainable out of {total_params:,} total")

    # Calculate warmup steps
    warmup_steps = len(train_batches) // 2  # Warmup for half an epoch

    # Train model with checkpointing
    print("\n=== Training Enhanced Character Transformer Model ===")
    model, (train_losses, val_losses) = train_model_with_checkpointing(
        model=model,
        train_batches=train_batches,
        val_batches=val_batches,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        device=device,
        patience=patience,
        label_smoothing=label_smoothing,
        gradient_accumulation_steps=gradient_accumulation_steps,
        use_mixed_precision=use_mixed_precision,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq,
        resume_from=args.resume_from
    )

    # Visualize final results
    visualize_results(train_losses, val_losses, 'final_training_results.png')
    print("\nTraining visualization saved to final_training_results.png")

    # Generate some text
    print("\n=== Generating Text ===")
    prompt = "The quick brown fox"
    generated_text = model.generate(
        prompt=prompt,
        max_length=200,
        temperature=0.6,
        top_k=50,
        top_p=0.92,
        repetition_penalty=1.2,
        tokenizer=tokenizer,
        device=device
    )
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")

    # Save final model
    torch.save(model.state_dict(), 'final_char_transformer_model.pt')
    print("Final model saved to final_char_transformer_model.pt")

if __name__ == "__main__":
    main() 