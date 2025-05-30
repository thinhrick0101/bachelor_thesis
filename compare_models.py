import torch
import torch.nn as nn
from pathlib import Path
import logging
import math
from torch.utils.data import DataLoader
from train_sparse_transformer import TextDataset, compute_bpb

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('logs/model_comparison.log'),
        logging.StreamHandler()
    ]
)

def load_model(checkpoint_path, device):
    """Load a saved model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'config' in checkpoint:  # Dense model format
        from char_transformer import CharTransformer
        config = checkpoint['config']
        model = CharTransformer(**config)
    else:  # Sparse model format
        from sparse_transformer_v2 import SparseTransformer
        model = SparseTransformer(
            vocab_size=256,
            d_model=512,
            nhead=8,
            num_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
            max_seq_length=1024
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model

def evaluate_model(model, data_loader, criterion, device, model_type=""):
    """Evaluate model on given dataset."""
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)
            
            output = model(data)
            output = output.view(-1, output.size(-1))
            target = target.view(-1)
            
            loss = criterion(output, target)
            
            total_loss += loss.item() * batch_size
            total_tokens += batch_size * data.size(1)
    
    avg_loss = total_loss / total_tokens
    bpb = compute_bpb(avg_loss)
    ppl = math.exp(avg_loss)
    
    logging.info(f"{model_type} Model Performance:")
    logging.info(f"- Bits per byte: {bpb:.4f}")
    logging.info(f"- Perplexity: {ppl:.2f}")
    logging.info(f"- Loss: {avg_loss:.4f}")
    
    return {
        'bpb': bpb,
        'ppl': ppl,
        'loss': avg_loss
    }

def compare_models(dense_path, sparse_path, test_data_path, batch_size=32, seq_length=1024):
    """Compare dense and sparse transformer models."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load test dataset
    test_dataset = TextDataset(test_data_path, seq_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Setup criterion
    criterion = nn.CrossEntropyLoss()
    
    # Load and evaluate dense model
    logging.info("\nEvaluating Dense Transformer...")
    dense_model = load_model(dense_path, device)
    dense_results = evaluate_model(dense_model, test_loader, criterion, device, "Dense")
    
    # Load and evaluate sparse model
    logging.info("\nEvaluating Sparse Transformer...")
    sparse_model = load_model(sparse_path, device)
    sparse_results = evaluate_model(sparse_model, test_loader, criterion, device, "Sparse")
    
    # Compare results
    bpb_diff = sparse_results['bpb'] - dense_results['bpb']
    ppl_diff = sparse_results['ppl'] - dense_results['ppl']
    
    logging.info("\nPerformance Comparison (Sparse - Dense):")
    logging.info(f"- Bits per byte difference: {bpb_diff:+.4f}")
    logging.info(f"- Perplexity difference: {ppl_diff:+.2f}")
    
    # Calculate parameter counts
    dense_params = sum(p.numel() for p in dense_model.parameters())
    sparse_params = sum(p.numel() for p in sparse_model.parameters())
    param_reduction = (1 - sparse_params/dense_params) * 100
    
    logging.info("\nModel Size Comparison:")
    logging.info(f"- Dense model parameters: {dense_params:,}")
    logging.info(f"- Sparse model parameters: {sparse_params:,}")
    logging.info(f"- Parameter reduction: {param_reduction:.1f}%")
    
    return {
        'dense': dense_results,
        'sparse': sparse_results,
        'differences': {
            'bpb': bpb_diff,
            'ppl': ppl_diff,
            'param_reduction': param_reduction
        }
    }

if __name__ == '__main__':
    dense_model_path = 'models/dense_transformer/best_model.pt'
    sparse_model_path = 'models/sparse_transformer/best_model.pt'
    test_data_path = 'data/enwik8/test.txt'
    
    results = compare_models(dense_model_path, sparse_model_path, test_data_path) 