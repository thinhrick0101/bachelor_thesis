#!/bin/bash

# Hyper-parameter Sensitivity Analysis Pipeline
# Complete workflow for testing robustness of sparse attention parameters

set -e

echo "🔬 HYPER-PARAMETER SENSITIVITY ANALYSIS"
echo "========================================"
echo

# Check if in correct directory
if [[ ! -f "benchmark_grid.py" ]]; then
    echo "❌ Error: benchmark_grid.py not found. Please run from bachelor_thesis directory."
    exit 1
fi

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Create output directory
mkdir -p sensitivity_analysis

# Ask user for test type
echo "Choose analysis type:"
echo "1) Quick test (3 configurations, ~30 minutes)"
echo "2) Full grid search (9 configurations, ~2 hours)"
echo "3) Custom grid (specify your own)"
echo

read -p "Enter choice (1/2/3): " choice

case $choice in
    1)
        echo "🚀 Running quick sensitivity test..."
        python benchmark_grid.py \
            --quick_test \
            --num_epochs 2 \
            --batch_size 8 \
            --seq_length 256 \
            --output_dir sensitivity_analysis/quick_test
        
        results_file="sensitivity_analysis/quick_test/grid_search_results_*.csv"
        ;;
    2)
        echo "🚀 Running full grid search..."
        python benchmark_grid.py \
            --num_epochs 3 \
            --batch_size 16 \
            --seq_length 512 \
            --output_dir sensitivity_analysis/full_grid
        
        results_file="sensitivity_analysis/full_grid/grid_search_results_*.csv"
        ;;
    3)
        echo "Enter custom grid as JSON (e.g., [[8,8,64],[16,8,64],[32,8,64]]):"
        read -p "Grid: " custom_grid
        
        echo "🚀 Running custom grid search..."
        python benchmark_grid.py \
            --custom_grid "$custom_grid" \
            --num_epochs 3 \
            --batch_size 16 \
            --seq_length 512 \
            --output_dir sensitivity_analysis/custom_grid
        
        results_file="sensitivity_analysis/custom_grid/grid_search_results_*.csv"
        ;;
    *)
        echo "❌ Invalid choice. Exiting."
        exit 1
        ;;
esac

echo
echo "⏳ Grid search completed. Processing results..."

# Find the most recent results file
results_csv=$(ls -t $results_file 2>/dev/null | head -1)

if [[ ! -f "$results_csv" ]]; then
    echo "❌ Error: No results file found. Grid search may have failed."
    exit 1
fi

echo "📊 Results file: $results_csv"

# Create visualizations
echo "🎨 Creating visualizations..."
python visualize_sensitivity_analysis.py "$results_csv" --output_dir "${results_csv%/*}/plots"

# Summary
echo
echo "✅ SENSITIVITY ANALYSIS COMPLETE!"
echo "=================================="
echo
echo "📁 Results directory: $(dirname "$results_csv")"
echo "📊 CSV results: $results_csv"
echo "🎨 Plots directory: $(dirname "$results_csv")/plots"
echo
echo "📋 Key files for thesis:"
echo "  • $(dirname "$results_csv")/plots/sensitivity_robustness_summary.png"
echo "  • $(dirname "$results_csv")/plots/sensitivity_thesis_table.csv"
echo "  • $(dirname "$results_csv")/plots/sensitivity_thesis_table.tex"
echo "  • $(dirname "$results_csv")/plots/robustness_analysis.txt"
echo
echo "💡 Next steps:"
echo "  1. Review robustness_analysis.txt for thesis text"
echo "  2. Include sensitivity_robustness_summary.png in your thesis"
echo "  3. Copy the table from sensitivity_thesis_table.csv"
echo "  4. Add the LaTeX table from sensitivity_thesis_table.tex"
echo

# Quick preview of results
if command -v head >/dev/null 2>&1; then
    echo "📋 Quick preview of results:"
    echo "=========================="
    head -10 "$results_csv" | column -t -s ","
    echo
fi

echo "🎉 Analysis pipeline completed successfully!" 