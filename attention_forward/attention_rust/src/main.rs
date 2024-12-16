fn attention_forward(
    out: &mut [f32],
    preatt: &mut [f32],
    att: &mut [f32],
    inp: &[f32],
    b: usize,
    t: usize,
    c: usize,
    nh: usize,
) {
    let c3 = c * 3;
    let hs = c / nh; // head size
    let scale = 1.0 / (hs as f32).sqrt();
    for b_idx in 0..b {
        for t_idx in 0..t {
            for h in 0..nh {
                let query_start = b_idx * t * c3 + t_idx * c3 + h * hs;
                let preatt_start = b_idx * nh * t * t + h * t * t + t_idx * t;
                let att_start = b_idx * nh * t * t + h * t * t + t_idx * t;

                // Pass 1: calculate query dot key and maxval
                let mut maxval = f32::NEG_INFINITY;
                for t2 in 0..=t_idx {
                    let key_start = b_idx * t * c3 + t2 * c3 + h * hs + c; // +c because it's key

                    // (query_t) dot (key_t2)
                    let mut val = 0.0;
                    for i in 0..hs {
                        val += inp[query_start + i] * inp[key_start + i];
                    }
                    val *= scale;
                    if val > maxval {
                        maxval = val;
                    }

                    preatt[preatt_start + t2] = val;
                }

                // Pass 2: calculate the exp and keep track of sum
                let mut expsum = 0.0;
                for t2 in 0..=t_idx {
                    let expv = (preatt[preatt_start + t2] - maxval).exp().min(f32::MAX);
                    expsum += expv;
                    att[att_start + t2] = expv;
                }
                let expsum_inv = if expsum == 0.0 { 0.0 } else { 1.0 / expsum };

                // Pass 3: normalize to get the softmax
                for t2 in 0..t {
                    if t2 <= t_idx {
                        att[att_start + t2] *= expsum_inv;
                    } else {
                        // causal attention mask, set to zero
                        att[att_start + t2] = 0.0;
                    }
                }

                // Pass 4: accumulate weighted values into the output of attention
                let out_start = b_idx * t * c + t_idx * c + h * hs;
                for i in 0..hs {
                    out[out_start + i] = 0.0;
                }
                for t2 in 0..=t_idx {
                    let value_start = b_idx * t * c3 + t2 * c3 + h * hs + c * 2; // +c*2 because it's value
                    let att_val = att[att_start + t2];
                    for i in 0..hs {
                        out[out_start + i] += att_val * inp[value_start + i];
                    }
                }
            }
        }
    }
}

fn main() {
    // Define dimensions
    let b = 1; // Batch size
    let t = 4; // Sequence length
    let c = 8; // Total hidden size
    let nh = 2; // Number of heads
    let hs = c / nh; // Head size
    let c3 = c * 3; // Size of concatenated Q, K, V

    // Allocate memory
    let inp_size = b * t * c3;
    let out_size = b * t * c;
    let att_size = b * nh * t * t;

    let mut inp = vec![0.0f32; inp_size];
    let mut out = vec![0.0f32; out_size];
    let mut preatt = vec![0.0f32; att_size];
    let mut att = vec![0.0f32; att_size];

    // Initialize input tensor 'inp' with sequential values
    for i in 0..inp_size {
        inp[i] = (i + 1) as f32 / 10.0; // Values like 0.1, 0.2, ...
    }

    // Call the attention_forward function
    attention_forward(&mut out, &mut preatt, &mut att, &inp, b, t, c, nh);

    // Print the results
    println!("Input (inp):");
    for b_idx in 0..b {
        for t_idx in 0..t {
            print!("Batch {}, Time {}: ", b_idx, t_idx);
            for c_idx in 0..c3 {
                let idx = b_idx * t * c3 + t_idx * c3 + c_idx;
                print!("{:.2} ", inp[idx]);
            }
            println!();
        }
    }

    println!("\nOutput (out):");
    for b_idx in 0..b {
        for t_idx in 0..t {
            print!("Batch {}, Time {}: ", b_idx, t_idx);
            for c_idx in 0..c {
                let idx = b_idx * t * c + t_idx * c + c_idx;
                print!("{:.4} ", out[idx]);
            }
            println!();
        }
    }

    println!("\nAttention Weights (att) per head:");
    for b_idx in 0..b {
        for h in 0..nh {
            println!("Batch {}, Head {}:", b_idx, h);
            for t_idx in 0..t {
                print!("Time {} -> ", t_idx);
                for t2 in 0..t {
                    let idx = b_idx * nh * t * t + h * t * t + t_idx * t + t2;
                    print!("{:.4} ", att[idx]);
                }
                println!();
            }
        }
    }
}

/* C Code

Output (out):
Batch 0, Time 0: 1.7000 1.8000 1.9000 2.0000 2.1000 2.2000 2.3000 2.4000
Batch 0, Time 1: 4.1000 4.2000 4.3000 4.4000 4.5000 4.6000 4.7000 4.8000
Batch 0, Time 2: 6.5000 6.6000 6.7000 6.8000 6.9000 7.0000 7.1000 7.2000
Batch 0, Time 3: 8.9000 9.0000 9.1000 9.2000 9.3000 9.4000 9.5000 9.6000

Attention Weights (att) per head:
Batch 0, Head 0:
Time 0 -> 1.0000 0.0000 0.0000 0.0000
Time 1 -> 0.0000 1.0000 0.0000 0.0000
Time 2 -> 0.0000 0.0000 1.0000 0.0000
Time 3 -> 0.0000 0.0000 0.0000 1.0000
Batch 0, Head 1:
Time 0 -> 1.0000 0.0000 0.0000 0.0000
Time 1 -> 0.0000 1.0000 0.0000 0.0000
Time 2 -> 0.0000 0.0000 1.0000 0.0000
Time 3 -> 0.0000 0.0000 0.0000 1.0000

Rust Code
Output (out):
Batch 0, Time 0: 1.7000 1.8000 1.9000 2.0000 2.1000 2.2000 2.3000 2.4000
Batch 0, Time 1: 4.1000 4.2000 4.3000 4.4000 4.5000 4.6000 4.7000 4.8000
Batch 0, Time 2: 6.5000 6.6000 6.7000 6.8000 6.9000 7.0000 7.1000 7.2000
Batch 0, Time 3: 8.9000 9.0000 9.1000 9.2000 9.3000 9.4000 9.5000 9.6000

Attention Weights (att) per head:
Batch 0, Head 0:
Time 0 -> 1.0000 0.0000 0.0000 0.0000
Time 1 -> 0.0000 1.0000 0.0000 0.0000
Time 2 -> 0.0000 0.0000 1.0000 0.0000
Time 3 -> 0.0000 0.0000 0.0000 1.0000
Batch 0, Head 1:
Time 0 -> 1.0000 0.0000 0.0000 0.0000
Time 1 -> 0.0000 1.0000 0.0000 0.0000
Time 2 -> 0.0000 0.0000 1.0000 0.0000
Time 3 -> 0.0000 0.0000 0.0000 1.0000
*/