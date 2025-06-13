#!/bin/bash
# launch_experiments_parallel.sh - Launch statistical experiments in parallel

echo "Starting parallel statistical analysis experiments..."
echo "This will launch 6 training runs (3 dense + 3 sparse) with different seeds in parallel"

# Configuration
SEEDS="111 222 333"
NUM_EPOCHS=50
BATCH_SIZE=32
SEQ_LENGTH=1024
LEARNING_RATE=1e-4

# Create models directory
mkdir -p bachelor_thesis/models

# Function to run dense model
run_dense() {
    local seed=$1
    echo "Starting dense model with seed $seed..."
    python bachelor_thesis/train_dense_model.py \
        --seed $seed \
        --num_epochs $NUM_EPOCHS \
        --wandb_run_name "dense_seed_$seed" \
        --batch_size $BATCH_SIZE \
        --seq_length $SEQ_LENGTH \
        --learning_rate $LEARNING_RATE > "dense_seed_${seed}.log" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✓ Dense model seed $seed completed successfully"
    else
        echo "✗ Dense model seed $seed failed (check dense_seed_${seed}.log)"
    fi
}

# Function to run sparse model
run_sparse() {
    local seed=$1
    echo "Starting sparse model with seed $seed..."
    python bachelor_thesis/train_sparse_transformer.py \
        --seed $seed \
        --num_epochs $NUM_EPOCHS \
        --wandb_run_name "sparse_seed_$seed" \
        --batch_size $BATCH_SIZE \
        --seq_length $SEQ_LENGTH \
        --learning_rate $LEARNING_RATE > "sparse_seed_${seed}.log" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✓ Sparse model seed $seed completed successfully"
    else
        echo "✗ Sparse model seed $seed failed (check sparse_seed_${seed}.log)"
    fi
}

echo "================================================================"
echo "LAUNCHING ALL EXPERIMENTS IN PARALLEL"
echo "================================================================"
echo "Note: Output is redirected to log files (e.g., dense_seed_111.log)"
echo "Monitor GPU usage with: watch -n 1 nvidia-smi"

# Launch all experiments in background
pids=()

for seed in $SEEDS; do
    run_dense $seed &
    pids+=($!)
    
    run_sparse $seed &
    pids+=($!)
done

echo "Launched ${#pids[@]} parallel training jobs"
echo "Waiting for all jobs to complete..."

# Wait for all background jobs to finish
for pid in ${pids[@]}; do
    wait $pid
done

echo "================================================================"
echo "ALL PARALLEL EXPERIMENTS COMPLETED"
echo "================================================================"
echo "Check individual log files for detailed output:"
for seed in $SEEDS; do
    echo "- dense_seed_${seed}.log"
    echo "- sparse_seed_${seed}.log"
done
echo ""
echo "Now run: python bachelor_thesis/aggregate_wandb_table.py"
echo "to generate the statistical summary table for your thesis." 