#!/bin/bash

export OMP_NUM_THREADS=4
export LDFLAGS="-L/opt/homebrew/opt/llvm/lib"
export CPPFLAGS="-I/opt/homebrew/opt/llvm/include"

/opt/homebrew/opt/llvm/bin/clang -O2 -fopenmp $LDFLAGS $CPPFLAGS  -o matmul_example matmul_example.c

echo "Run"
./matmul_example 64 1024 768 768
wait
./matmul_example 64 1024 768 768
wait
./matmul_example 64 1024 768 768
wait

