#!/bin/bash
cargo clean
export OPENBLAS_NUM_THREADS=8  # Adjust based on your CPU

RUSTFLAGS="-C target-cpu=native" cargo build --release
./target/release/blas_matmul --b 2 --t 3 --c 4 --oc 5

./target/release/blas_matmul --b 64 --t 1024 --c 768 --oc 768
wait
./target/release/blas_matmul --b 64 --t 1024 --c 768 --oc 768
wait
./target/release/blas_matmul --b 64 --t 1024 --c 768 --oc 768
wait