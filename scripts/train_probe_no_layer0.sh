#!/bin/bash
# Train carry probe excluding layer 0 to force learning computational patterns

# Configuration
LAYERS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35"
OUTPUT_DIR="runs/carry_probe/no_layer0_$(date +%Y%m%d_%H%M%S)"

echo "Training probe WITHOUT Layer 0 (embeddings)"
echo "This forces the probe to find computational patterns in deeper layers"
echo ""
echo "Configuration:"
echo "  Layers: ${LAYERS}"
echo "  Token position: answer"
echo "  L1 penalty: 1e-5 (for sparsity)"
echo "  L2 penalty: 1e-4 (for regularization)"
echo "  Epochs: 30"
echo "  Batch size: 64"
echo ""

sbatch scripts/sbatch_run.sh python scripts/train_carry_probe.py \
    --layers ${LAYERS} \
    --token_position answer \
    --l1_penalty 1e-5 \
    --l2_penalty 1e-4 \
    --max_value 99 \
    --n_epochs 10 \
    --batch_size 64 \
    --learning_rate 5e-3 \
    --early_stopping_patience 5 \
    --save_epochs \
    --output_dir ${OUTPUT_DIR}

echo ""
echo "Training complete. Results saved to: ${OUTPUT_DIR}"
