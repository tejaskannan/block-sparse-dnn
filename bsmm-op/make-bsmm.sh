#!/bin/bash

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

nvcc -std=c++11 -c -o bsmm.cu.o bsmm.cc.cu \
  ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr

g++ -std=c++11 -shared -o bsmm.so bsmm.cc \
  bsmm.cu.o ${TF_CFLAGS[@]} -fPIC -L/usr/local/cuda/lib64 -lcudart ${TF_LFLAGS[@]} -fpermissive
