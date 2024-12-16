#!/bin/bash
cargo clean
#export OPENBLAS_NUM_THREADS=16

RUSTFLAGS="-C target-cpu=native" cargo build --release
./target/release/llm

# sudo cargo flamegraph --release