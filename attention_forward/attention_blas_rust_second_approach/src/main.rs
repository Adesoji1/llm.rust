use cblas::{sgemm, Layout, Transpose};

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

                // Extract the query vector Q: shape (hs)
                let query_vec = &inp[query_start..query_start + hs]; // shape: hs

                // Construct keys_mat: (t_idx+1) x hs
                let mut keys_mat = Vec::with_capacity((t_idx + 1) * hs);
                for t2 in 0..=t_idx {
                    let key_start = b_idx * t * c3 + t2 * c3 + h * hs + c; // +c for keys
                    keys_mat.extend_from_slice(&inp[key_start..key_start + hs]);
                }

                // We'll now compute preatt_row = keys_mat * query_vec
                // keys_mat: (t_idx+1) x hs
                // query_vec: hs x 1
                // result = preatt_row: (t_idx+1) x 1
                let mut preatt_row = vec![0.0f32; t_idx + 1];

                unsafe {
                    sgemm(
                        Layout::RowMajor,
                        Transpose::None,   // A as is: (t_idx+1) x hs
                        Transpose::None,   // B as is: hs x 1
                        (t_idx + 1) as i32, // m
                        1,                  // n
                        hs as i32,          // k
                        1.0,                // alpha
                        &keys_mat,          // A
                        hs as i32,          // lda = hs
                        query_vec,          // B
                        1,                  // ldb = 1 (because B is hs x 1)
                        0.0,                // beta
                        &mut preatt_row,    // C
                        1,                  // ldc = 1
                    );
                }

                // Apply scaling
                for val in &mut preatt_row {
                    *val *= scale;
                }

                // Softmax over preatt_row[0..=t_idx]
                let maxval = preatt_row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

                let mut expsum = 0.0;
                for t2 in 0..=t_idx {
                    let expv = (preatt_row[t2] - maxval).exp().min(f32::MAX);
                    expsum += expv;
                    att[att_start + t2] = expv;
                    preatt[preatt_start + t2] = preatt_row[t2];
                }

                let expsum_inv = if expsum == 0.0 { 0.0 } else { 1.0 / expsum };

                for t2 in 0..t {
                    if t2 <= t_idx {
                        att[att_start + t2] *= expsum_inv;
                    } else {
                        att[att_start + t2] = 0.0;
                    }
                }

                let out_start = b_idx * t * c + t_idx * c + h * hs;
                for i in 0..hs {
                    out[out_start + i] = 0.0;
                }

                // Construct values_mat: (t_idx+1) x hs
                let att_row = &att[att_start..att_start + (t_idx + 1)];
                let mut values_mat = Vec::with_capacity((t_idx + 1) * hs);
                for t2 in 0..=t_idx {
                    let value_start = b_idx * t * c3 + t2 * c3 + h * hs + c * 2;
                    values_mat.extend_from_slice(&inp[value_start..value_start + hs]);
                }

                let mut out_vec = vec![0.0f32; hs];

                // Now out_vec = att_row(1 x (t_idx+1)) * values_mat((t_idx+1) x hs)
                // Dimensions for sgemm:
                // A = att_row: (1 x (t_idx+1))
                // B = values_mat: ((t_idx+1) x hs)
                // C = out_vec: (1 x hs)
                // M=1, N=hs, K=(t_idx+1)
                unsafe {
                    sgemm(
                        Layout::RowMajor,
                        Transpose::None,   // A is (1x(t_idx+1))
                        Transpose::None,   // B is ((t_idx+1)xhs)
                        1,                 // M=1
                        hs as i32,         // N=hs
                        (t_idx + 1) as i32, // K=(t_idx+1)
                        1.0,               // alpha
                        att_row,           // A, lda=(t_idx+1)
                        (t_idx + 1) as i32,
                        &values_mat,       // B, ldb=hs
                        hs as i32,
                        0.0,               // beta
                        &mut out_vec,      // C, ldc=hs
                        hs as i32,
                    );
                }

                out[out_start..out_start + hs].copy_from_slice(&out_vec);
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

    // Call the attention_forward function that uses cblas
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
