#!/bin/bash


/opt/homebrew/opt/llvm/bin/clang -O2 -o layernorm_forward main.c

echo "Run"
#./matmul_backward 2 3 4 5
./layernorm_forward
wait
