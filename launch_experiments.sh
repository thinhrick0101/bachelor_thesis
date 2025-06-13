#!/bin/bash
# launch_experiments.sh - Launch statistical experiments for thesis

echo "Starting statistical analysis experiments..."
echo "This will launch 6 training runs (3 dense + 3 sparse) with different seeds"

# Configuration
SEEDS="111 222 333"
NUM_EPOCHS=50
BATCH_SIZE=32
SEQ_LENGTH=1024
LEARNING_RATE=1e-4

# Create models directory
mkdir -p bachelor_thesis/models

echo "================================================================"
echo "LAUNCHING DENSE MODEL EXPERIMENTS"
echo "================================================================"

for seed in $SEEDS; do
    echo "Starting dense model with seed $seed..."
    python bachelor_thesis/train_dense_model.py \
        --seed $seed \
        --num_epochs $NUM_EPOCHS \
        --wandb_run_name "dense_seed_$seed" \
        --batch_size $BATCH_SIZE \
        --seq_length $SEQ_LENGTH \
        --learning_rate $LEARNING_RATE
    
    if [ $? -eq 0 ]; then
        echo "✓ Dense model seed $seed completed successfully"
    else
        echo "✗ Dense model seed $seed failed"
    fi
    echo "----------------------------------------"
done

echo "================================================================"
echo "LAUNCHING SPARSE MODEL EXPERIMENTS"
echo "================================================================"

for seed in $SEEDS; do
    echo "Starting sparse model with seed $seed..."
    python bachelor_thesis/train_sparse_transformer.py \
        --seed $seed \
        --num_epochs $NUM_EPOCHS \
        --wandb_run_name "sparse_seed_$seed" \
        --batch_size $BATCH_SIZE \
        --seq_length $SEQ_LENGTH \
        --learning_rate $LEARNING_RATE
    
    if [ $? -eq 0 ]; then
        echo "✓ Sparse model seed $seed completed successfully"
    else
        echo "✗ Sparse model seed $seed failed"
    fi
    echo "----------------------------------------"
done

echo "================================================================"
echo "ALL EXPERIMENTS COMPLETED"
echo "================================================================"
echo "Now run: python bachelor_thesis/aggregate_wandb_table.py"
echo "to generate the statistical summary table for your thesis." 