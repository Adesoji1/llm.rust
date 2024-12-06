#!/bin/bash

# Arrays for each parameter
b_values=(4 8 16 32 64 128 256 512 1024)
t_values=(64 128 256 512 1024 2048 4096 8192 16384)
c_oc_values=(48 96 192 384 768 1536 3072 6144 12288)

# Loop through all combinations
for ((i=0; i<6; i++)); do
    b=${b_values[i]}
    t=${t_values[i]}
    c=${c_oc_values[i]}
    oc=${c_oc_values[i]}
    echo "b: $b, t: $t, c: $c, oc: $oc"
    ./target/release/blas_matmul --b $b --t $t --c $c --oc $oc
    wait
    ./target/release/blas_matmul --b $b --t $t --c $c --oc $oc
    wait
    ./target/release/blas_matmul --b $b --t $t --c $c --oc $oc
    wait
    done
