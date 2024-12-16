use cblas::{sgemm, Layout, Transpose};


fn attention_forward_blas(
    out: &mut [f32],
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
        for h in 0..nh {
            // Allocate arrays
            let mut q = vec![0.0f32; t * hs];
            let mut k = vec![0.0f32; t * hs];
            let mut v = vec![0.0f32; t * hs];
            let mut s = vec![0.0f32; t * t];
            let mut attn = vec![0.0f32; t * t];
            let mut out_head = vec![0.0f32; t * hs];

            // Fill q, k, v
            for t_idx in 0..t {
                let q_start = b_idx * t * c3 + t_idx * c3 + h * hs;
                let k_start = b_idx * t * c3 + t_idx * c3 + c + h * hs;
                let v_start = b_idx * t * c3 + t_idx * c3 + c * 2 + h * hs;
                for i in 0..hs {
                    q[t_idx * hs + i] = inp[q_start + i];
                    k[t_idx * hs + i] = inp[k_start + i];
                    v[t_idx * hs + i] = inp[v_start + i];
                }
            }

            // Compute s = scale * q * k^T
            unsafe {
                sgemm(
                    Layout::RowMajor,
                    Transpose::None,
                    Transpose::Ordinary,
                    t as i32,  // m
                    t as i32,  // n
                    hs as i32, // k
                    scale,
                    &q,
                    hs as i32,
                    &k,
                    hs as i32,
                    0.0,
                    &mut s,
                    t as i32,
                );
            }

            // Apply causal mask
            for i in 0..t {
                for j in i + 1..t {
                    s[i * t + j] = f32::NEG_INFINITY;
                }
            }

            // Softmax computation
            for i in 0..t {
                let mut maxval = f32::NEG_INFINITY;
                for j in 0..=i {
                    let val = s[i * t + j];
                    if val > maxval {
                        maxval = val;
                    }
                }
                let mut expsum = 0.0;
                for j in 0..=i {
                    let expv = (s[i * t + j] - maxval).exp();
                    attn[i * t + j] = expv;
                    expsum += expv;
                }
                for j in 0..=i {
                    attn[i * t + j] /= expsum;
                }
            }

            // Compute out_head = attn * v
            unsafe {
                sgemm(
                    Layout::RowMajor,
                    Transpose::None,
                    Transpose::None,
                    t as i32,  // m
                    hs as i32, // n
                    t as i32,  // k
                    1.0,
                    &attn,
                    t as i32,
                    &v,
                    hs as i32,
                    0.0,
                    &mut out_head,
                    hs as i32,
                );
            }

            // Write out_head to out
            for t_idx in 0..t {
                let out_start = b_idx * t * c + t_idx * c + h * hs;
                for i in 0..hs {
                    out[out_start + i] = out_head[t_idx * hs + i];
                }
            }

            // Store attn in att array
            let attn_start = b_idx * nh * t * t + h * t * t;
            for i in 0..t {
                for j in 0..t {
                    att[attn_start + i * t + j] = attn[i * t + j];
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
    let mut att_scores = vec![0.0f32; att_size];

    // Initialize input tensor 'inp' with sequential values
    for i in 0..inp_size {
        inp[i] = (i + 1) as f32 / 10.0; // Values like 0.1, 0.2, ...
    }

    // Call the attention_forward function
    attention_forward_blas(&mut out,&mut att_scores, &inp, b, t, c, nh);

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

    // Updated printing code
    println!("\nAttention Weights (att) per head:");
    for b_idx in 0..b {
        for h in 0..nh {
            println!("Batch {}, Head {}:", b_idx, h);
            for t_idx in 0..t {
                print!("Time {} -> ", t_idx);
                for t2 in 0..t {
                    let idx = b_idx * nh * t * t + h * t * t + t_idx * t + t2;
                    print!("{:.4} ", att_scores[idx]);
                }
                println!();
            }
        }
    }
}
