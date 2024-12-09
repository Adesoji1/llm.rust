#!/bin/bash
cargo clean
RUSTFLAGS="-C target-cpu=native" cargo build --release
./target/release/llm

# sudo cargo flamegraph --release