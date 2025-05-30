import torch
import matplotlib.pyplot as plt
import math
import numpy as np
from pathlib import Path

def load_training_metrics(checkpoint_path):
    """Load training metrics from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return {
        'train_loss': checkpoint.get('train_loss', None),
        'val_loss': checkpoint.get('val_loss', None),
        'train_bpb': checkpoint.get('train_bpb', None),
        'val_bpb': checkpoint.get('val_bpb', None),
        'epoch': checkpoint.get('epoch', None)
    }

def plot_training_metrics(metrics_list, save_dir='plots'):
    """Plot training metrics over time."""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    epochs = range(len(metrics_list))
    
    # Extract metrics
    train_losses = [m['train_loss'] for m in metrics_list]
    val_losses = [m['val_loss'] for m in metrics_list]
    train_bpb = [m['train_bpb'] for m in metrics_list]
    val_bpb = [m['val_bpb'] for m in metrics_list]
    train_ppl = [math.exp(loss) for loss in train_losses]
    val_ppl = [math.exp(loss) for loss in val_losses]
    
    # Plot Loss
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(save_dir / 'loss_plot.png')
    plt.close()
    
    # Plot Bits per Byte
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_bpb, label='Train BPB', marker='o')
    plt.plot(epochs, val_bpb, label='Validation BPB', marker='o')
    plt.title('Training and Validation Bits per Byte')
    plt.xlabel('Epoch')
    plt.ylabel('Bits per Byte')
    plt.grid(True)
    plt.legend()
    plt.savefig(save_dir / 'bpb_plot.png')
    plt.close()
    
    # Plot Perplexity
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_ppl, label='Train Perplexity', marker='o')
    plt.plot(epochs, val_ppl, label='Validation Perplexity', marker='o')
    plt.title('Training and Validation Perplexity')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.grid(True)
    plt.legend()
    plt.yscale('log')  # Use log scale for perplexity
    plt.savefig(save_dir / 'perplexity_plot.png')
    plt.close()
    
    # Print summary statistics
    print("\nTraining Summary:")
    print(f"Final train loss: {train_losses[-1]:.4f}")
    print(f"Final validation loss: {val_losses[-1]:.4f}")
    print(f"Best validation loss: {min(val_losses):.4f}")
    print(f"\nFinal train BPB: {train_bpb[-1]:.4f}")
    print(f"Final validation BPB: {val_bpb[-1]:.4f}")
    print(f"Best validation BPB: {min(val_bpb):.4f}")
    print(f"\nFinal train perplexity: {train_ppl[-1]:.2f}")
    print(f"Final validation perplexity: {val_ppl[-1]:.2f}")
    print(f"Best validation perplexity: {min(val_ppl):.2f}")

def main():
    # Load all checkpoints
    checkpoint_dir = Path('models/sparse_transformer')
    checkpoint_files = sorted(checkpoint_dir.glob('checkpoint_epoch_*.pt'))
    
    if not checkpoint_files:
        print("No checkpoint files found!")
        return
    
    # Load metrics from all checkpoints
    metrics_list = []
    for checkpoint_file in checkpoint_files:
        metrics = load_training_metrics(checkpoint_file)
        if all(v is not None for v in metrics.values()):
            metrics_list.append(metrics)
    
    if not metrics_list:
        print("No valid metrics found in checkpoints!")
        return
    
    # Plot metrics
    plot_training_metrics(metrics_list)
    print("\nPlots have been saved in the 'plots' directory.")

if __name__ == '__main__':
    main() 