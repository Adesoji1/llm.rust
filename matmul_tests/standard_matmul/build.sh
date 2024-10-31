#!/bin/bash
cargo clean
cargo build --release
./target/release/standard_matmul --b 64 --t 1024 --c 768 --oc 768
wait
./target/release/standard_matmul --b 64 --t 1024 --c 768 --oc 768
wait
./target/release/standard_matmul --b 64 --t 1024 --c 768 --oc 768
wait