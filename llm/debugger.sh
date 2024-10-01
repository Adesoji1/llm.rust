#!/bin/bash
cargo clean
RUSTFLAGS=-g cargo build --release
lldb target/release/llm