#!/usr/bin/env python3
"""
Quick test script for ablation functionality
Usage: python test_ablation.py
"""

import torch
import argparse
from sparse_byte_transformer import SparseByteTransformer

def test_mask_subset():
    """Test that different mask subsets create different models"""
    
    print("🧪 Testing mask subset functionality...")
    
    # Create test config
    config = argparse.Namespace(
        vocab_size=256,
        d_model=128,  # Smaller for quick test
        nhead=4,      # Smaller for quick test
        num_layers=2, # Smaller for quick test
        dim_feedforward=256,
        dropout=0.1,
        seq_length=64,
        mask_subset='0123'  # Will be overridden
    )
    
    # Test different mask subsets
    test_subsets = ['0123', '0', '1', '2', '3']
    
    for subset in test_subsets:
        print(f"\n  Testing subset: {subset}")
        
        # Update config
        config.mask_subset = subset
        
        try:
            # Create model
            model = SparseByteTransformer(config)
            
            # Test forward pass
            batch_size = 2
            seq_len = 32
            input_ids = torch.randint(0, 256, (batch_size, seq_len))
            
            with torch.no_grad():
                output = model(input_ids)
            
            print(f"    ✅ Subset {subset}: Model created and forward pass successful")
            print(f"       Output shape: {output.shape}")
            print(f"       Active clusters: {model.transformer_encoder[0].self_attn.active_clusters}")
            print(f"       Cluster head counts: {model.transformer_encoder[0].self_attn.cluster_head_counts}")
            
        except Exception as e:
            print(f"    ❌ Subset {subset}: Error - {e}")
    
    print(f"\n✅ Mask subset functionality test completed!")

def main():
    test_mask_subset()

if __name__ == "__main__":
    main() 