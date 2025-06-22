#!/usr/bin/env python3
"""
Hyper-parameter sensitivity analysis for sparse attention patterns.
Evaluates validation perplexity, training speed, and memory usage across
a grid of window size (w), stride (s), and global anchors (a) parameters.
"""

import os
import sys
import json
import time
import argparse
import subprocess
import itertools
from pathlib import Path
from datetime import datetime
import torch
import numpy as np
import pandas as pd

# Configuration grid for sensitivity analysis
DEFAULT_GRID = [
    (8, 4, 64),   # Small window, small stride
    (8, 8, 64),   # Small window, medium stride  
    (8, 16, 64),  # Small window, large stride
    (16, 4, 64),  # Medium window, small stride
    (16, 8, 64),  # BASELINE: Medium window, medium stride
    (16, 16, 64), # Medium window, large stride
    (32, 4, 64),  # Large window, small stride
    (32, 8, 64),  # Large window, medium stride
    (32, 16, 64), # Large window, large stride
]

class GridSearchManager:
    """Manages grid search experiments for sparse attention parameters."""
    
    def __init__(self, output_dir='sensitivity_analysis', num_epochs=5, 
                 batch_size=32, seq_length=1024, learning_rate=1e-4):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Training configuration
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.learning_rate = learning_rate
        
        # Results tracking
        self.results = []
        self.start_time = datetime.now()
        
        # Create results CSV
        self.results_file = self.output_dir / f"grid_search_results_{self.start_time.strftime('%Y%m%d_%H%M%S')}.csv"
        
    def create_sparse_config(self, window_size, stride, anchors):
        """Create sparse attention configuration."""
        return {
            'vocab_size': 256,
            'd_model': 512,
            'nhead': 8,
            'num_layers': 6,  # Reduced for faster experimentation
            'dim_feedforward': 2048,
            'dropout': 0.1,
            'seq_length': self.seq_length,
            'window_size': window_size,
            'stride': stride,
            'global_anchors': anchors,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs
        }
    
    def run_single_experiment(self, window_size, stride, anchors):
        """Run a single experiment with given parameters."""
        tag = f"w{window_size}_s{stride}_a{anchors}"
        exp_dir = self.output_dir / tag
        exp_dir.mkdir(exist_ok=True)
        
        print(f"\n🔄 Running experiment: {tag}")
        print(f"   Window: {window_size}, Stride: {stride}, Anchors: {anchors}")
        
        # Create configuration
        config = self.create_sparse_config(window_size, stride, anchors)
        config_file = exp_dir / "config.json"
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Track memory and time
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        start_time = time.time()
        
        try:
            # Run training experiment
            result = self._run_training_experiment(config, exp_dir, tag)
            
            # Calculate metrics
            end_time = time.time()
            training_time = end_time - start_time
            peak_memory = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
            
            # Store results
            experiment_result = {
                'window_size': window_size,
                'stride': stride,
                'global_anchors': anchors,
                'tag': tag,
                'val_ppl': result.get('val_ppl', float('inf')),
                'train_ppl': result.get('train_ppl', float('inf')),
                'tokens_per_sec': result.get('tokens_per_sec', 0),
                'peak_memory_mb': peak_memory,
                'training_time_sec': training_time,
                'converged': result.get('converged', False),
                'final_loss': result.get('final_loss', float('inf'))
            }
            
            self.results.append(experiment_result)
            
            # Save intermediate results
            self.save_results()
            
            print(f"✅ Completed {tag}: PPL={result.get('val_ppl', 'N/A'):.3f}, "
                  f"Tok/s={result.get('tokens_per_sec', 0):.0f}, "
                  f"Mem={peak_memory:.0f}MB")
            
            return experiment_result
            
        except Exception as e:
            print(f"❌ Failed {tag}: {e}")
            # Still record the failure
            failed_result = {
                'window_size': window_size,
                'stride': stride,
                'global_anchors': anchors,
                'tag': tag,
                'val_ppl': float('inf'),
                'train_ppl': float('inf'),
                'tokens_per_sec': 0,
                'peak_memory_mb': 0,
                'training_time_sec': 0,
                'converged': False,
                'final_loss': float('inf'),
                'error': str(e)
            }
            self.results.append(failed_result)
            self.save_results()
            return failed_result
    
    def _run_training_experiment(self, config, exp_dir, tag):
        """Run the actual training experiment."""
        # Import training functions
        try:
            from sparse_byte_transformer import SparseByteTransformer
            from train_sparse_model import train_sparse_model
            from byte_dataset import load_enwik8_data, create_batches
        except ImportError as e:
            print(f"Import error: {e}")
            # Try alternative imports
            try:
                sys.path.append(str(Path(__file__).parent))
                from sparse_byte_transformer import SparseByteTransformer
            except ImportError:
                raise ImportError("Could not import required modules")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model with custom sparse parameters
        model_config = type('Config', (), config)()
        model = SparseByteTransformer(model_config).to(device)
        
        # Load data (subset for speed)
        try:
            # Try to load existing preprocessed data
            train_data = torch.load('data/enwik8_splits/train.bin')
            val_data = torch.load('data/enwik8_splits/val.bin')
            
            # Use subset for grid search (faster evaluation)
            subset_size = min(len(train_data), 100000)  # Limit data size
            train_data = train_data[:subset_size]
            val_data = val_data[:min(len(val_data), 10000)]
            
        except FileNotFoundError:
            # Fallback: create dummy data for testing
            print("Warning: Using dummy data for grid search")
            train_data = torch.randint(0, 256, (50000,))
            val_data = torch.randint(0, 256, (5000,))
        
        # Create batches
        train_batches = create_batches(train_data, config['batch_size'], config['seq_length'])
        val_batches = create_batches(val_data, config['batch_size'], config['seq_length'])
        
        # Limit number of batches for faster experiments
        max_train_batches = 100
        max_val_batches = 20
        
        train_batches = train_batches[:max_train_batches]
        val_batches = val_batches[:max_val_batches]
        
        print(f"   Using {len(train_batches)} train batches, {len(val_batches)} val batches")
        
        # Track timing
        tokens_processed = 0
        start_time = time.time()
        
        # Training with minimal epochs for speed
        train_losses, val_losses = train_sparse_model(
            model=model,
            train_batches=train_batches,
            val_batches=val_batches,
            num_epochs=config['num_epochs'],
            learning_rate=config['learning_rate'],
            device=device,
            save_path=str(exp_dir / f"{tag}_model.pt")
        )
        
        end_time = time.time()
        
        # Calculate metrics
        training_time = end_time - start_time
        tokens_processed = len(train_batches) * config['batch_size'] * config['seq_length'] * config['num_epochs']
        tokens_per_sec = tokens_processed / training_time if training_time > 0 else 0
        
        # Get final metrics
        final_train_loss = train_losses[-1] if train_losses else float('inf')
        final_val_loss = val_losses[-1] if val_losses else float('inf')
        
        # Calculate perplexity
        train_ppl = np.exp(final_train_loss) if final_train_loss != float('inf') else float('inf')
        val_ppl = np.exp(final_val_loss) if final_val_loss != float('inf') else float('inf')
        
        return {
            'val_ppl': val_ppl,
            'train_ppl': train_ppl,
            'tokens_per_sec': tokens_per_sec,
            'converged': len(val_losses) >= config['num_epochs'],
            'final_loss': final_val_loss
        }
    
    def save_results(self):
        """Save current results to CSV."""
        if self.results:
            df = pd.DataFrame(self.results)
            df.to_csv(self.results_file, index=False)
            print(f"💾 Results saved to {self.results_file}")
    
    def run_grid_search(self, grid=None):
        """Run complete grid search."""
        if grid is None:
            grid = DEFAULT_GRID
        
        print(f"🚀 Starting grid search with {len(grid)} configurations")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"⏱️  Estimated time: ~{len(grid) * 10} minutes")
        
        baseline_idx = None
        for i, (w, s, a) in enumerate(grid):
            if w == 16 and s == 8:  # Find baseline
                baseline_idx = i
                break
        
        # Run baseline first for reference
        if baseline_idx is not None:
            print(f"\n🎯 Running BASELINE first: {grid[baseline_idx]}")
            baseline_result = self.run_single_experiment(*grid[baseline_idx])
            
            # Run remaining experiments
            remaining_grid = [config for i, config in enumerate(grid) if i != baseline_idx]
        else:
            remaining_grid = grid
        
        for window_size, stride, anchors in remaining_grid:
            self.run_single_experiment(window_size, stride, anchors)
        
        # Final results summary
        self.print_summary()
        return self.results
    
    def print_summary(self):
        """Print summary of results."""
        if not self.results:
            print("No results to summarize")
            return
        
        df = pd.DataFrame(self.results)
        
        print(f"\n{'='*80}")
        print("📊 HYPER-PARAMETER SENSITIVITY ANALYSIS SUMMARY")
        print(f"{'='*80}")
        
        # Find baseline
        baseline = df[(df['window_size'] == 16) & (df['stride'] == 8)]
        if not baseline.empty:
            baseline_ppl = baseline.iloc[0]['val_ppl']
            baseline_speed = baseline.iloc[0]['tokens_per_sec']
            baseline_mem = baseline.iloc[0]['peak_memory_mb']
            
            print(f"🎯 BASELINE (w=16, s=8, a=64):")
            print(f"   Val PPL: {baseline_ppl:.3f}")
            print(f"   Speed: {baseline_speed:.0f} tok/s")
            print(f"   Memory: {baseline_mem:.0f} MB")
            print()
        
        # Summary table
        print("📋 FULL RESULTS TABLE:")
        print("Configuration (w,s,a)      | Val PPL ↓ | Tok/s ↑ | Mem (MB) | Status")
        print("-" * 70)
        
        for _, row in df.iterrows():
            w, s, a = row['window_size'], row['stride'], row['global_anchors']
            ppl = row['val_ppl']
            speed = row['tokens_per_sec']
            mem = row['peak_memory_mb']
            status = "✅" if row['converged'] else "❌"
            
            baseline_tag = " (BASELINE)" if w == 16 and s == 8 else ""
            
            if ppl == float('inf'):
                ppl_str = "FAILED"
            else:
                ppl_str = f"{ppl:.3f}"
            
            print(f"({w:2d},{s:2d},{a:2d}){baseline_tag:<12} | {ppl_str:>8} | {speed:>7.0f} | {mem:>7.0f} | {status}")
        
        # Analysis
        valid_results = df[df['val_ppl'] != float('inf')]
        if not valid_results.empty:
            print(f"\n📈 ANALYSIS:")
            ppl_range = valid_results['val_ppl'].max() - valid_results['val_ppl'].min()
            speed_range = valid_results['tokens_per_sec'].max() - valid_results['tokens_per_sec'].min()
            
            print(f"   PPL variation: {ppl_range:.3f} (robustness: {'✅ GOOD' if ppl_range <= 0.03 else '⚠️  HIGH'})")
            print(f"   Speed variation: {speed_range:.0f} tok/s")
            
            # Best configurations
            best_ppl_idx = valid_results['val_ppl'].idxmin()
            best_speed_idx = valid_results['tokens_per_sec'].idxmax()
            
            best_ppl_config = valid_results.loc[best_ppl_idx]
            best_speed_config = valid_results.loc[best_speed_idx]
            
            print(f"\n🏆 BEST CONFIGURATIONS:")
            print(f"   Best PPL: (w={best_ppl_config['window_size']}, s={best_ppl_config['stride']}, a={best_ppl_config['global_anchors']}) → {best_ppl_config['val_ppl']:.3f}")
            print(f"   Best Speed: (w={best_speed_config['window_size']}, s={best_speed_config['stride']}, a={best_speed_config['global_anchors']}) → {best_speed_config['tokens_per_sec']:.0f} tok/s")

def create_batches(data, batch_size, seq_length):
    """Create batches from data tensor."""
    # Simple batching for grid search
    num_batches = len(data) // (batch_size * seq_length)
    batches = []
    
    for i in range(num_batches):
        start_idx = i * batch_size * seq_length
        end_idx = start_idx + batch_size * seq_length
        batch_data = data[start_idx:end_idx].reshape(batch_size, seq_length)
        batches.append(batch_data)
    
    return batches

def main():
    """Main entry point for grid search."""
    parser = argparse.ArgumentParser(description='Sparse attention hyper-parameter sensitivity analysis')
    parser.add_argument('--output_dir', type=str, default='sensitivity_analysis',
                        help='Output directory for results')
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='Number of epochs per experiment (reduced for speed)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (reduced for speed)')
    parser.add_argument('--seq_length', type=int, default=512,
                        help='Sequence length (reduced for speed)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--custom_grid', type=str, default=None,
                        help='Custom grid as JSON string: "[[8,4,64],[16,8,64]]"')
    parser.add_argument('--quick_test', action='store_true',
                        help='Run quick test with 3 configurations only')
    
    args = parser.parse_args()
    
    # Setup grid
    if args.custom_grid:
        grid = json.loads(args.custom_grid)
    elif args.quick_test:
        grid = [(8, 8, 64), (16, 8, 64), (32, 8, 64)]  # Quick test
    else:
        grid = DEFAULT_GRID
    
    # Create and run grid search
    manager = GridSearchManager(
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        learning_rate=args.learning_rate
    )
    
    results = manager.run_grid_search(grid)
    
    print(f"\n🎉 Grid search completed!")
    print(f"📁 Results saved in: {manager.output_dir}")
    print(f"📊 CSV file: {manager.results_file}")

if __name__ == "__main__":
    main() 