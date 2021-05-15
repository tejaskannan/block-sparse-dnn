#!/bin/bash

for script in "bsmm_bench.py" "tfbs_bench.py"
do
    for matrix_size in 64 256 1024 2048
    do
        for block_size in 4 8 16 32
        do
            for sparsityDenom in 10 20 50 100
            do
                python $script $matrix_size $block_size $(($(($matrix_size * $matrix_size))/$(($block_size * $block_size * $sparsityDenom)))) 100 > DATA$script$matrix_size$block_size$sparsityDenom
            done
        done
    done
done