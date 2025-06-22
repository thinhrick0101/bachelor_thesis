#!/usr/bin/env python3
"""
Inference benchmarking script for ByteTransformer models.
Measures latency, throughput, GPU utilization, and memory usage.

Usage:
    python benchmark_inference.py --ckpt sparse.pt --batch 1 --iters 200
    python benchmark_inference.py --ckpt dense.pt --batch 8 --iters 200
"""

import argparse
import time
import torch
import csv
import os
import signal
import sys
import numpy as np
from pathlib import Path

# Try to import pynvml for GPU monitoring
try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    print("Warning: pynvml not available. GPU utilization monitoring disabled.")
    print("Install with: pip install nvidia-ml-py3")
    HAS_PYNVML = False

# Import model classes
try:
    from sparse_byte_transformer import SparseByteTransformer
    from byte_transformer import EnhancedCharTransformer
    HAS_MODELS = True
except ImportError:
    print("Warning: Model classes not found. Trying alternate imports...")
    try:
        from stable_char_transformer import EnhancedCharTransformer
        HAS_MODELS = True
    except ImportError:
        print("Error: Could not import model classes")
        HAS_MODELS = False


class ByteTokenizer:
    """Simple byte-level tokenizer for benchmarking."""
    
    def __init__(self):
        self.vocab_size = 256
    
    def encode(self, text):
        """Encode text to byte-level tokens."""
        if isinstance(text, str):
            return list(text.encode('utf-8'))
        return text
    
    def decode(self, tokens):
        """Decode byte-level tokens to text."""
        return bytes(tokens).decode('utf-8', errors='ignore')


def load_enwik8_data(data_path='data/enwik8', seq_length=1024):
    """Load a sample of enwik8 data for benchmarking."""
    
    # Try different possible paths
    possible_paths = [
        data_path,
        'data/enwik8',
        '../data/enwik8',
        'bachelor_thesis/data/enwik8'
    ]
    
    enwik8_path = None
    for path in possible_paths:
        if os.path.exists(path):
            enwik8_path = path
            break
    
    if enwik8_path is None:
        # Create dummy data for benchmarking
        print("Warning: enwik8 data not found. Using dummy data for benchmarking.")
        # Create realistic text-like data
        dummy_text = "The quick brown fox jumps over the lazy dog. " * 100
        return dummy_text.encode('utf-8')
    
    # Load real data
    with open(enwik8_path, 'rb') as f:
        data = f.read()
    
    return data


def get_eval_batch(batch_size, seq_length=1024, device='cuda'):
    """Generate evaluation batch of the specified size."""
    
    # Load or create data
    data = load_enwik8_data(seq_length=seq_length)
    
    # Sample random sequences
    batch_data = []
    max_start = len(data) - seq_length - 1
    
    for _ in range(batch_size):
        if max_start > 0:
            start_idx = np.random.randint(0, max_start)
            sequence = data[start_idx:start_idx + seq_length]
        else:
            # If data is too short, repeat it
            sequence = (data * ((seq_length // len(data)) + 1))[:seq_length]
        
        # Convert to list of integers (byte values)
        batch_data.append(list(sequence))
    
    # Convert to tensor
    tensor = torch.tensor(batch_data, dtype=torch.long, device=device)
    return tensor


def load_model_from_checkpoint(ckpt_path, device='cuda'):
    """Load model from checkpoint, detecting model type."""
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Detect model type and configuration
    if 'sparse' in ckpt_path.lower():
        # Sparse model configuration
        config = type('Config', (), {
            'vocab_size': 256,
            'd_model': 512,
            'nhead': 8,
            'num_layers': 12,
            'dim_feedforward': 2048,
            'dropout': 0.1,
            'seq_length': 1024,
            'mask_subset': '0123',
            'tie_weights': True,
            'scale_embeddings': True,
            'token_dropout': 0.0
        })()
        
        model = SparseByteTransformer(config)
    else:
        # Dense model configuration
        config = {
            'vocab_size': 256,
            'd_model': 512,
            'nhead': 8,
            'num_layers': 12,
            'dim_feedforward': 2048,
            'dropout': 0.1,
            'attention_dropout': 0.1,
            'activation_dropout': 0.1,
            'token_dropout': 0.05,
            'use_checkpoint': False,  # Disable for inference
            'stochastic_depth_prob': 0.1
        }
        
        model = EnhancedCharTransformer(**config)
    
    # Load state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model.to(device).eval()


def start_gpu_monitor():
    """Start GPU utilization monitoring in a separate process."""
    if not HAS_PYNVML:
        return None
    
    pid = os.fork()
    if pid == 0:  # Child process
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            with open('/tmp/gpu_util.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'gpu_util', 'mem_util', 'mem_used_mb'])
                
                while True:
                    try:
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        writer.writerow([
                            time.time(), 
                            util.gpu, 
                            util.memory, 
                            mem.used // (1024 * 1024)  # Convert to MB
                        ])
                        f.flush()
                        time.sleep(0.05)  # Sample every 50ms
                    except:
                        break
        except:
            pass
        finally:
            os._exit(0)
    
    return pid


def stop_gpu_monitor(pid):
    """Stop GPU monitoring and return statistics."""
    if pid is None or not HAS_PYNVML:
        return 0.0, 0
    
    try:
        os.kill(pid, signal.SIGTERM)
        os.waitpid(pid, 0)
    except:
        pass
    
    # Read utilization data
    try:
        with open('/tmp/gpu_util.csv', 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            if rows:
                gpu_util = sum(float(row['gpu_util']) for row in rows) / len(rows)
                max_mem = max(int(row['mem_used_mb']) for row in rows)
                return gpu_util, max_mem
    except:
        pass
    
    return 0.0, 0


def timed_run(model, batch_size, iters, seq_length=1024):
    """Run timed inference benchmark."""
    
    device = next(model.parameters()).device
    
    # Warm-up to fill caches & trigger kernel compilation
    print(f"Warming up with {batch_size} batch size...")
    with torch.inference_mode():
        dummy = get_eval_batch(batch_size, seq_length, device)
        for _ in range(10):
            _ = model(dummy)
    
    torch.cuda.synchronize()
    
    # Prepare for timing
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    
    total_tokens = batch_size * seq_length * iters
    
    # Start GPU monitoring
    monitor_pid = start_gpu_monitor()
    
    # Get evaluation batch
    inp = get_eval_batch(batch_size, seq_length, device)
    
    # Benchmark proper
    print(f"Running {iters} iterations...")
    with torch.inference_mode():
        start_evt.record()
        for i in range(iters):
            if i % 50 == 0:
                print(f"  Iteration {i}/{iters}")
            _ = model(inp)
        end_evt.record()
    
    torch.cuda.synchronize()
    
    # Stop monitoring and get stats
    gpu_util, max_mem = stop_gpu_monitor(monitor_pid)
    
    # Calculate metrics
    elapsed_ms = start_evt.elapsed_time(end_evt)
    latency_ms = elapsed_ms / iters
    throughput_tokens_per_sec = total_tokens / (elapsed_ms / 1000)
    
    return latency_ms, throughput_tokens_per_sec, gpu_util, max_mem


def main():
    parser = argparse.ArgumentParser(description='Benchmark inference performance')
    parser.add_argument('--ckpt', required=True, help='Path to model checkpoint')
    parser.add_argument('--batch', type=int, default=1, help='Batch size')
    parser.add_argument('--iters', type=int, default=200, help='Number of iterations')
    parser.add_argument('--seq_length', type=int, default=1024, help='Sequence length')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile')
    args = parser.parse_args()
    
    if not HAS_MODELS:
        print("Error: Required model classes not available")
        sys.exit(1)
    
    print(f"Loading model from {args.ckpt}...")
    try:
        model = load_model_from_checkpoint(args.ckpt)
        print(f"Model loaded successfully. Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        if args.compile:
            print("Compiling model with torch.compile...")
            model = torch.compile(model)
        
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    print(f"Benchmarking with batch_size={args.batch}, iters={args.iters}, seq_length={args.seq_length}")
    
    try:
        latency, throughput, gpu_util, max_mem = timed_run(
            model, args.batch, args.iters, args.seq_length
        )
        
        # Print results in CSV format
        model_name = Path(args.ckpt).stem
        print(f"\nResults:")
        print(f"Model,Batch,Latency (ms),Throughput (tok/s),GPU Util (%),Peak Mem (MB)")
        print(f"{model_name},{args.batch},{latency:.2f},{throughput:.1f},{gpu_util:.1f},{max_mem}")
        
        # Also print human-readable format
        print(f"\nDetailed Results:")
        print(f"  Model: {model_name}")
        print(f"  Batch size: {args.batch}")
        print(f"  Latency per sequence: {latency:.2f} ms")
        print(f"  Throughput: {throughput:.1f} tokens/second")
        print(f"  GPU utilization: {gpu_util:.1f}%")
        print(f"  Peak memory: {max_mem} MB")
        
    except Exception as e:
        print(f"Error during benchmarking: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 