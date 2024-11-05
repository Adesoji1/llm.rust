#!/bin/bash
cargo clean
cargo build --release
./target/release/rayon_matmul --b 2 --t 3 --c 4 --oc 5
# ./target/release/standard_matmul --b 64 --t 1024 --c 768 --oc 768
# wait
# ./target/release/standard_matmul --b 64 --t 1024 --c 768 --oc 768
# wait
# ./target/release/standard_matmul --b 64 --t 1024 --c 768 --oc 768
# wait