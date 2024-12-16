#!/bin/bash
cargo clean
export OPENBLAS_NUM_THREADS=8  # Adjust based on your CPU

RUSTFLAGS="-C target-cpu=native" cargo build --release
./target/release/attention_blas_rust_second_approach
wait