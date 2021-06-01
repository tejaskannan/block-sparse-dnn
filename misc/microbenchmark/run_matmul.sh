#!/bin/sh

HIDDEN_SIZE=1024
TRIALS=150
SPARSITY=0.0625

echo "===== Dense Matmul ====="
python dense_matmul.py --hidden-size ${HIDDEN_SIZE} --sparsity ${SPARSITY} --trials ${TRIALS}

echo "===== Sparse Matmul ====="
python sparse_matmul.py --hidden-size ${HIDDEN_SIZE} --sparsity ${SPARSITY} --trials ${TRIALS}

echo "===== Block Sparse Matmul ====="
python bsparse_matmul.py --hidden-size ${HIDDEN_SIZE} --sparsity ${SPARSITY} --trials ${TRIALS} --block-sizes 8 16 32
