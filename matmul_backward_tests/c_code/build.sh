#!/bin/bash

export OMP_NUM_THREADS=4
export LDFLAGS="-L/opt/homebrew/opt/llvm/lib"
export CPPFLAGS="-I/opt/homebrew/opt/llvm/include"

/opt/homebrew/opt/llvm/bin/clang -O2 -fopenmp $LDFLAGS $CPPFLAGS  -o matmul_backward matmul_back.c

echo "Run"
./matmul_backward 2 3 4 5
# ./matmul_backward 64 1024 768 768
# wait
# ./matmul_backward 64 1024 768 768
# wait
# ./matmul_backward 64 1024 768 768
# wait

