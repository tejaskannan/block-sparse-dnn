#!/bin/bash

for script in "tfbs_bench.py" "openai_bench.py"
do
    for matrix_size in 64 256 512 
    do
        for block_size in 8 16 32
        do
            for sparsity in "0.1" "0.05" "0.02" "0.01"
            do
                python $script $matrix_size $block_size $sparsity 1000 > DATA$script$matrix_size$block_size$sparsity
            done
        done
    done
done
