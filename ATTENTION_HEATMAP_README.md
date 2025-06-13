# Attention Heatmap Figure Generation for Thesis RQ5

This directory contains tools to generate the **one-page attention heat-map figure** for Research Question 5 (RQ5) on interpretability, following the provided recipe.

## 📊 What This Generates

A single PNG figure with **four subplots**, each showing a representative attention head from discovered clusters:

- **(a) Cluster-0: Focused-Local** - Short-range local attention patterns
- **(b) Cluster-1: Strided** - Strided patterns every N tokens  
- **(c) Cluster-2: Global-Anchor** - Global attention with anchor points
- **(d) Cluster-3: Wider-Local** - Medium-range local attention

## 🚀 Quick Start

### Method 1: Automatic (Recommended)
```bash
# Automatically finds trained model and generates figure
./bachelor_thesis/generate_thesis_figure.sh
```

### Method 2: Manual
```bash
# Specify exact model path
./bachelor_thesis/generate_thesis_figure.sh models/dense_seed_111.pt

# Or run Python script directly
python bachelor_thesis/extract_attention_heatmaps.py \
    --model_path models/dense_seed_111.pt \
    --data_path data/enwik8 \
    --seq_length 512 \
    --output figs/attention_clusters_heatmap.png
```

## 📁 Generated Files

After running, you'll get:

```
figs/
├── attention_clusters_heatmap.png     # Main figure for thesis
├── attention_clusters_heatmap.tex     # LaTeX caption 
└── attention_clusters_heatmap.md      # Markdown caption
```

## 📝 Integration with Thesis

### 1. Copy Figure
```bash
cp figs/attention_clusters_heatmap.png /path/to/thesis/figures/
```

### 2. LaTeX Integration
```latex
\begin{figure}[h]
  \centering
  \includegraphics[width=\linewidth]{figures/attention_clusters_heatmap.png}
  \caption{Representative self-attention heads discovered in the dense model.
  Each heat-map shows attention probability (query row $\rightarrow$ key column).
  \textbf{(a)} Cluster-0 focused-local; \textbf{(b)} Cluster-1 strided;
  \textbf{(c)} Cluster-2 global-anchor; \textbf{(d)} Cluster-3 wider-local.
  These patterns were used as fixed masks in the sparse Transformer.}
  \label{fig:cluster_heatmaps}
\end{figure}
```

### 3. Reference in Text
```latex
Figure \ref{fig:cluster_heatmaps} visualizes the four empirically 
discovered attention patterns. The focused-local head (a) behaves like 
a 16-token sliding window, whereas the strided head (b) leaps every 8 
positions, etc. These interpretable masks confirm that dense heads 
specialize and motivate our fixed sparse design.
```

## 🔧 How It Works

### 1. Attention Extraction
- Uses forward hooks to capture attention weights from each transformer layer
- Modifies `_attention_block` temporarily to extract weights with `need_weights=True`
- Collects attention matrices for all heads across all layers

### 2. Cluster Assignment
The script automatically identifies representative heads using pattern analysis:

- **Focused-Local**: High diagonal bias + low entropy
- **Strided**: High stride strength (periodic patterns)
- **Global-Anchor**: Low sparsity + high entropy  
- **Wider-Local**: Balanced entropy and sparsity

### 3. Visualization
- Creates 2×2 subplot layout
- Uses `magma` colormap for better visibility
- Downsamples large attention matrices for readability
- Adds proper labels, colorbars, and annotations

## ⚙️ Configuration Options

### Command Line Arguments
```bash
python bachelor_thesis/extract_attention_heatmaps.py \
    --model_path MODEL_PATH \          # Path to trained dense model
    --data_path data/enwik8 \           # Validation data path
    --seq_length 512 \                  # Sequence length for analysis
    --output attention_heatmap.png      # Output figure name
```

### Model Requirements
- **Dense Model**: Must be trained EnhancedCharTransformer
- **Architecture**: 12 layers, 8 heads, 512 dimensions
- **Format**: PyTorch `.pt` checkpoint with `model_state_dict`

## 🐛 Troubleshooting

### Common Issues

**1. No attention patterns extracted**
```bash
# Check model architecture matches
# Ensure model has 'model_state_dict' key in checkpoint
```

**2. Model file not found**
```bash
# List available models
find . -name "*.pt" -type f

# Use specific path
./generate_thesis_figure.sh models/your_model.pt
```

**3. Data loading errors**
```bash
# Script will use random sequence as fallback
# Ensure data/enwik8 exists or specify different path
```

**4. Memory issues**
```bash
# Reduce sequence length
python extract_attention_heatmaps.py --seq_length 256
```

### Expected Patterns

The generated heatmaps should show:
- **Focused-Local**: Strong diagonal band
- **Strided**: Regular periodic patterns  
- **Global-Anchor**: Vertical/horizontal lines (attending to special positions)
- **Wider-Local**: Broader diagonal band than focused-local

## 📊 Validation

To verify the figure is correct:

1. **Visual Check**: Each subplot should show distinct patterns
2. **Head Assignment**: Different (layer, head) combinations for each cluster
3. **File Size**: PNG should be ~200-500KB at 300 DPI
4. **Caption**: LaTeX file should contain proper references

## 🔄 Alternative Usage

### For Different Models
```bash
# Sparse model (if you want to compare)
python extract_attention_heatmaps.py \
    --model_path models/sparse_seed_111.pt \
    --output figs/sparse_attention_heatmap.png
```

### Custom Sequence
```bash
# Use specific text sequence
echo "Your custom text here" > custom_text.txt
python extract_attention_heatmaps.py \
    --data_path custom_text.txt \
    --seq_length 256
```

## 📖 Related Files

- `stable_char_transformer.py` - Model architecture
- `cluster_attention_patterns.py` - Detailed cluster analysis
- `attention_analysis.py` - General attention analysis tools
- `sparse_attention.py` - Sparse attention implementation

## 🎯 Thesis Integration Checklist

- [ ] Generated figure shows 4 distinct attention patterns
- [ ] Each subplot has clear (a), (b), (c), (d) labels
- [ ] LaTeX caption includes actual layer/head numbers
- [ ] Figure resolution is 300 DPI for print quality
- [ ] File copied to thesis figures directory
- [ ] Referenced properly in RQ5 discussion
- [ ] Used to motivate sparse attention design choices

This figure directly addresses RQ5's interpretability requirement by showing how dense attention heads naturally specialize into distinct, interpretable patterns that can guide sparse attention design. 