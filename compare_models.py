import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import logging
import math
from torch.utils.data import DataLoader, Dataset
from train_sparse_transformer import compute_bpb
from stable_char_transformer import ByteTokenizer, load_data

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('logs/model_comparison.log'),
        logging.StreamHandler()
    ]
)

class TextDataset(Dataset):
    """Dataset for byte-level text data."""
    def __init__(self, data_path, seq_length):
        self.data = load_data(data_path)
        self.seq_length = seq_length
        self.tokenizer = ByteTokenizer()
        
        # Convert data to tensor of byte indices
        self.tokens = self.tokenizer.encode(self.data)
        
        # Calculate number of sequences
        self.num_sequences = len(self.tokens) - seq_length
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        # Get sequence and target
        sequence = self.tokens[idx:idx + self.seq_length]
        target = self.tokens[idx + 1:idx + self.seq_length + 1]
        return sequence, target

def load_model(checkpoint_path, device):
    """Load a saved model checkpoint."""
    logging.info(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Debug checkpoint structure
    logging.info(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    # Determine model type from checkpoint structure
    if 'transformer_blocks.0.gamma1' in checkpoint:  # Dense model format
        from stable_char_transformer import EnhancedTransformerBlock, EnhancedCharTransformer
        
        # Create custom attention class that matches the checkpoint format
        class LegacyAttention(nn.Module):
            def __init__(self, d_model, nhead, dropout=0.1):
                super().__init__()
                self.d_model = d_model
                self.nhead = nhead
                self.head_dim = d_model // nhead
                self.in_proj_weight = nn.Parameter(torch.empty(3 * d_model, d_model))
                self.in_proj_bias = nn.Parameter(torch.empty(3 * d_model))
                self.out_proj = nn.Linear(d_model, d_model)
                self.dropout = dropout
                self._reset_parameters()
            
            def _reset_parameters(self):
                nn.init.xavier_uniform_(self.in_proj_weight)
                nn.init.xavier_uniform_(self.out_proj.weight)
                nn.init.zeros_(self.in_proj_bias)
                nn.init.zeros_(self.out_proj.bias)
            
            def forward(self, query, key, value, attn_mask=None, need_weights=False):
                # Combined QKV projection
                qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
                q, k, v = qkv.chunk(3, dim=-1)
                
                # Reshape for attention
                batch_size, seq_len, _ = query.shape
                q = q.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
                k = k.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
                v = v.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
                
                # Compute attention
                scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
                
                # Handle attention mask
                if attn_mask is not None:
                    # Convert float mask to boolean mask
                    if attn_mask.dtype == torch.float32 or attn_mask.dtype == torch.float64:
                        bool_mask = (attn_mask == float('-inf')).bool()
                    else:
                        bool_mask = attn_mask.bool()
                    
                    # Ensure mask matches the sequence length
                    if bool_mask.size(0) != seq_len:
                        # Create a new mask of the correct size
                        bool_mask = torch.triu(torch.ones(seq_len, seq_len, device=attn_mask.device), diagonal=1).bool()
                    
                    # Apply mask to scores
                    scores = scores.masked_fill(bool_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
                
                attn = F.softmax(scores, dim=-1)
                attn = F.dropout(attn, p=self.dropout, training=self.training)
                
                # Apply attention to values
                x = torch.matmul(attn, v)
                x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
                x = self.out_proj(x)
                
                return x, attn if need_weights else None
        
        # Create model with config matching the checkpoint
        model = EnhancedCharTransformer(
            vocab_size=256,  # Byte-level vocab size
            d_model=512,
            nhead=8,
            num_layers=12,  # Based on transformer_blocks.11 in checkpoint
            dim_feedforward=2048,
            dropout=0.1,
            attention_dropout=0.1,
            activation_dropout=0.1,
            token_dropout=0.05,
            use_checkpoint=False,  # Disable for inference
            stochastic_depth_prob=0.1,
            attention_class=LegacyAttention
        )
        
        # Load state dict directly since it matches the model architecture
        model.load_state_dict(checkpoint)
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
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Try loading the entire checkpoint as state dict
            model.load_state_dict(checkpoint)
    
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
    dense_model_path = 'models/dense_transformer/dense_char_transformer.pt'
    sparse_model_path = 'models/sparse_transformer/best_model.pt'
    test_data_path = 'data/enwik8'
    
    results = compare_models(dense_model_path, sparse_model_path, test_data_path) 