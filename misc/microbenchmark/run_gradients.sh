#!/bin/sh

HIDDEN_SIZE=1024
TRIALS=150
SPARSITY=0.0625

echo "===== Dense Gradients ====="
python dense_grad.py --hidden-size ${HIDDEN_SIZE} --sparsity ${SPARSITY} --trials ${TRIALS}

echo "===== Sparse Gradients ====="
python sparse_grad.py --hidden-size ${HIDDEN_SIZE} --sparsity ${SPARSITY} --trials ${TRIALS}

echo "===== Block Sparse Gradients ====="
python bsparse_grad.py --hidden-size ${HIDDEN_SIZE} --sparsity ${SPARSITY} --trials ${TRIALS} --block-sizes 8 16 32
