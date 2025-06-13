#!/usr/bin/env python3
"""
validate_setup.py - Validate experiment setup before running

This script checks if all dependencies and configurations are correct
for running the statistical analysis experiments.
"""

import os
import sys
import importlib
import torch
import argparse

def check_python_version():
    """Check if Python version is adequate"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 7:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} (adequate)")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (requires >= 3.7)")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    print("\nChecking dependencies...")
    required_packages = [
        'torch', 'numpy', 'matplotlib', 'wandb', 'pandas'
    ]
    
    missing = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (missing)")
            missing.append(package)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    return True

def check_cuda():
    """Check CUDA availability"""
    print("\nChecking CUDA...")
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        memory = torch.cuda.get_device_properties(current_device).total_memory / 1e9
        
        print(f"✓ CUDA available")
        print(f"  - Devices: {device_count}")
        print(f"  - Current: {device_name}")
        print(f"  - Memory: {memory:.1f} GB")
        
        # Test basic CUDA operations
        try:
            x = torch.randn(100, 100).cuda()
            y = torch.mm(x, x.t())
            print(f"✓ CUDA operations working")
            return True
        except Exception as e:
            print(f"✗ CUDA operations failed: {e}")
            return False
    else:
        print("⚠ CUDA not available (will use CPU - much slower)")
        return True

def check_files():
    """Check if required files exist"""
    print("\nChecking required files...")
    
    required_files = [
        'bachelor_thesis/train_dense_model.py',
        'bachelor_thesis/train_sparse_transformer.py',
        'bachelor_thesis/stable_char_transformer.py',
        'bachelor_thesis/sparse_byte_transformer.py',
        'bachelor_thesis/aggregate_wandb_table.py'
    ]
    
    missing = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} (missing)")
            missing.append(file_path)
    
    if missing:
        print(f"\nMissing files: {', '.join(missing)}")
        return False
    return True

def check_data():
    """Check if data is available"""
    print("\nChecking data availability...")
    
    data_path = 'data/enwik8'
    if os.path.exists(data_path):
        size_mb = os.path.getsize(data_path) / 1e6
        print(f"✓ {data_path} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"⚠ {data_path} not found")
        print("  Data will be downloaded automatically during training")
        return True

def check_wandb():
    """Check WandB configuration"""
    print("\nChecking WandB setup...")
    
    try:
        import wandb
        # Try to initialize WandB (will prompt for login if needed)
        print("Testing WandB connection...")
        
        # Check if already logged in
        try:
            api = wandb.Api()
            user = api.viewer
            print(f"✓ WandB logged in as: {user['username']}")
            return True
        except Exception:
            print("⚠ WandB not logged in")
            print("  Run: wandb login")
            print("  Or set WANDB_API_KEY environment variable")
            return False
            
    except Exception as e:
        print(f"✗ WandB check failed: {e}")
        return False

def test_argument_parsing():
    """Test if argument parsing works correctly"""
    print("\nTesting argument parsing...")
    
    # Test dense model
    try:
        import sys
        old_argv = sys.argv
        sys.argv = ['train_dense_model.py', '--seed', '42', '--num_epochs', '1']
        
        # Import and test argument parsing
        sys.path.insert(0, 'bachelor_thesis')
        from train_dense_model import argparse
        
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--num_epochs', type=int, default=20)
        parser.add_argument('--wandb_run_name', type=str, default=None)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--seq_length', type=int, default=1024)
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        
        args = parser.parse_args()
        print(f"✓ Argument parsing works (seed={args.seed}, epochs={args.num_epochs})")
        
        sys.argv = old_argv
        return True
        
    except Exception as e:
        print(f"✗ Argument parsing failed: {e}")
        sys.argv = old_argv
        return False

def estimate_runtime():
    """Estimate experiment runtime"""
    print("\nEstimating runtime...")
    
    # Rough estimates based on typical hardware
    if torch.cuda.is_available():
        # GPU estimates
        epochs = 20
        minutes_per_epoch = 2  # Rough estimate for RTX 2080 Ti with our settings
        total_minutes = epochs * minutes_per_epoch * 6  # 6 runs total
        
        print(f"✓ Estimated total runtime: ~{total_minutes//60}h {total_minutes%60}m")
        print(f"  - Per run: ~{epochs * minutes_per_epoch}m ({epochs} epochs)")
        print(f"  - Sequential: ~{total_minutes//60}h {total_minutes%60}m")
        print(f"  - Parallel (if GPU memory allows): ~{(epochs * minutes_per_epoch)//60}h {(epochs * minutes_per_epoch)%60}m")
    else:
        print("⚠ CPU training will be much slower (10-50x)")
        print("  Consider using GPU for reasonable training times")

def main():
    print("="*60)
    print("EXPERIMENT SETUP VALIDATION")
    print("="*60)
    
    checks = [
        check_python_version(),
        check_dependencies(),
        check_cuda(),
        check_files(),
        check_data(),
        check_wandb(),
        test_argument_parsing()
    ]
    
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    passed = sum(checks)
    total = len(checks)
    
    if passed == total:
        print(f"✓ All checks passed ({passed}/{total})")
        print("\nReady to launch experiments!")
        print("\nNext steps:")
        print("1. Run: ./bachelor_thesis/launch_experiments.sh")
        print("   or: ./bachelor_thesis/launch_experiments_parallel.sh")
        print("2. Wait for completion (~2-4 hours)")
        print("3. Run: python bachelor_thesis/aggregate_wandb_table.py")
    else:
        print(f"⚠ {total-passed} checks failed ({passed}/{total} passed)")
        print("\nPlease fix the issues above before running experiments.")
    
    estimate_runtime()
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 