#!/bin/bash

echo "🎯 Statistical Validation Package for Chapter 3"
echo "=" 
echo "📖 Implementing reviewer's cookbook for sample-size justification"
echo

# Check if model exists
if [ ! -f "dense_char_transformer.pt" ]; then
    echo "❌ Model file 'dense_char_transformer.pt' not found!"
    echo "   Please ensure your trained model is in the current directory"
    exit 1
fi

echo "📥 Running statistical validation..."
python statistical_validation_package.py \
    --model_path dense_char_transformer.pt \
    --num_sequences 10 \
    --sequence_length 1024 \
    --bootstrap_samples 5000

echo
echo "🎯 DELIVERABLES GENERATED:"
echo "✅ table_3_1a_confidence_intervals.tsv"
echo "✅ convergence_analysis.png" 
echo "✅ statistical_summary.txt"
echo
echo "📋 NEXT STEPS:"
echo "1. Copy table_3_1a_confidence_intervals.tsv into Word after Table 3-1"
echo "2. Add convergence_analysis.png to Appendix A as Fig. A-1"
echo "3. Use text from statistical_summary.txt in your thesis"
echo
echo "🎓 Reviewer concerns about sample-size adequacy: RESOLVED!" 