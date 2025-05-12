import os
import torch
import time
import json

def save_checkpoint(model, optimizer, scheduler, epoch, train_losses, val_losses, 
                   best_val_loss, checkpoint_dir='checkpoints', filename=None):
    """
    Save model checkpoint with additional training state
    
    Args:
        model: The model to save
        optimizer: Optimizer state
        scheduler: Learning rate scheduler state
        epoch: Current epoch number
        train_losses: List of training losses
        val_losses: List of validation losses
        best_val_loss: Best validation loss so far
        checkpoint_dir: Directory to save checkpoints
        filename: Custom filename (if None, use timestamp)
    
    Returns:
        Path to the saved checkpoint
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Generate filename with timestamp if not provided
    if filename is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_epoch_{epoch}_{timestamp}.pt"
    
    # Full path to checkpoint file
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    
    # Save checkpoint with all training state
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss
    }
    
    # Add scheduler state if provided
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # Save the checkpoint
    torch.save(checkpoint, checkpoint_path)
    
    # Save the training metrics separately as JSON for easy analysis
    metrics = {
        'epoch': epoch,
        'train_losses': [float(loss) for loss in train_losses],
        'val_losses': [float(loss) for loss in val_losses] if val_losses else None,
        'best_val_loss': float(best_val_loss) if best_val_loss != float('inf') else None
    }
    
    metrics_path = os.path.join(checkpoint_dir, f"metrics_epoch_{epoch}.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Checkpoint saved to {checkpoint_path}")
    return checkpoint_path

def load_checkpoint(model, optimizer=None, scheduler=None, checkpoint_path=None, 
                   checkpoint_dir='checkpoints', device=None):
    """
    Load model checkpoint and training state
    
    Args:
        model: The model to load weights into
        optimizer: Optimizer to load state into (optional)
        scheduler: Learning rate scheduler to load state into (optional)
        checkpoint_path: Path to specific checkpoint file
        checkpoint_dir: Directory to look for latest checkpoint if path not provided
        device: Device to load the model onto
    
    Returns:
        tuple: (model, optimizer, scheduler, epoch, train_losses, val_losses, best_val_loss)
    """
    # If no specific checkpoint is provided, find the latest one
    if checkpoint_path is None and os.path.exists(checkpoint_dir):
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint') and f.endswith('.pt')]
        if checkpoint_files:
            # Sort by epoch and timestamp (assuming filename format: checkpoint_epoch_X_TIMESTAMP.pt)
            checkpoint_files.sort(key=lambda x: (int(x.split('_')[2]), x.split('_')[3]))
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[-1])
    
    # If no checkpoint found, return the original model and states
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print("No checkpoint found. Starting fresh training.")
        return model, optimizer, scheduler, 0, [], [], float('inf')
    
    # Determine device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move model to device
    model = model.to(device)
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Get training state
    epoch = checkpoint.get('epoch', 0)
    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    print(f"Resumed from epoch {epoch}")
    
    return model, optimizer, scheduler, epoch, train_losses, val_losses, best_val_loss

def find_latest_checkpoint(checkpoint_dir='checkpoints'):
    """
    Find the latest checkpoint in the given directory
    
    Args:
        checkpoint_dir: Directory to look for checkpoints
    
    Returns:
        Path to the latest checkpoint, or None if no checkpoints found
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint') and f.endswith('.pt')]
    if not checkpoint_files:
        return None
    
    # Sort by epoch and timestamp
    checkpoint_files.sort(key=lambda x: (int(x.split('_')[2]), x.split('_')[3]))
    return os.path.join(checkpoint_dir, checkpoint_files[-1]) 