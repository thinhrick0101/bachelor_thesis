#!/bin/bash
# generate_thesis_figure.sh - Generate attention heatmap figure for thesis

echo "🎨 Generating Attention Heatmap Figure for Thesis"
echo "================================================="

# Configuration
MODEL_PATH="bachelor_thesis/dense_char_transformer.pt"  # Or your trained dense model
DATA_PATH="data/enwik8"
SEQ_LENGTH=512
OUTPUT="attention_clusters_heatmap.png"

# Alternative model paths to try
ALTERNATIVE_MODELS=(
    "bachelor_thesis/dense_char_transformer.pt"
    "bachelor_thesis/models/dense_byte_transformer.pt"
    "models/enhanced_char_transformer_model.pt"
    "enhanced_char_transformer_model.pt"
    "char_transformer_model.pt"
    "stable_char_transformer_model.pt"
)

# Find available model
FOUND_MODEL=""
for model in "${ALTERNATIVE_MODELS[@]}"; do
    if [ -f "$model" ]; then
        FOUND_MODEL="$model"
        echo "✅ Found model: $model"
        break
    fi
done

if [ -z "$FOUND_MODEL" ]; then
    echo "❌ No trained dense model found. Please specify model path."
    echo "   Available files:"
    find . -name "*.pt" -type f 2>/dev/null | head -10
    echo ""
    echo "Usage: $0 [model_path]"
    echo "Example: $0 models/dense_seed_111.pt"
    exit 1
fi

# Use provided model path or found model
if [ ! -z "$1" ]; then
    MODEL_PATH="$1"
    echo "📁 Using provided model: $MODEL_PATH"
else
    MODEL_PATH="$FOUND_MODEL"
    echo "📁 Using found model: $MODEL_PATH"
fi

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model file not found: $MODEL_PATH"
    exit 1
fi

# Create output directory for figures
mkdir -p figs

# Run the extraction script
echo "🔍 Extracting attention patterns..."
python bachelor_thesis/simple_attention_extractor.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --seq_length $SEQ_LENGTH \
    --output "figs/$OUTPUT"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 SUCCESS! Generated files:"
    echo "   📊 Figure: figs/$OUTPUT"
    echo "   📝 LaTeX: figs/attention_clusters_heatmap.tex"
    echo "   📝 Markdown: figs/attention_clusters_heatmap.md"
    echo ""
    echo "📋 Next steps for thesis:"
    echo "1. Copy figs/$OUTPUT to your thesis figures directory"
    echo "2. Include the LaTeX caption from figs/attention_clusters_heatmap.tex"
    echo "3. Reference as \\ref{fig:cluster_heatmaps} in your text"
    echo ""
    echo "💡 Tip: View the generated figure to verify it shows distinct patterns!"
else
    echo "❌ Failed to generate attention heatmap figure"
    exit 1
fi 