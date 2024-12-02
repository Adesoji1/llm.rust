#!/bin/bash

# We should create a parallel Attention for Rust?

/opt/homebrew/opt/llvm/bin/clang -O2 -o attention_forward main.c

echo "Run"
./attention_forward
wait
