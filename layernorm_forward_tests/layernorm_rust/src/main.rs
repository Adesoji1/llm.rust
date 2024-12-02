fn layernorm_forward(
    out: &mut [f32],
    mean: &mut [f32],
    rstd: &mut [f32],
    inp: &[f32],
    weight: &[f32],
    bias: &[f32],
    b: usize,
    t: usize,
    c: usize,
) {
    let eps = 1e-5f32;

    for b_idx in 0..b {
        for t_idx in 0..t {
            let start_idx = b_idx * t * c + t_idx * c;
            let x = &inp[start_idx..start_idx + c];
            let out_bt = &mut out[start_idx..start_idx + c];

            // Calculate the mean
            let m: f32 = x.iter().sum::<f32>() / c as f32;

            // Calculate the variance (without any bias correction)
            let v: f32 = x.iter()
                .map(|&xi| {
                    let xshift = xi - m;
                    xshift * xshift
                })
                .sum::<f32>() / c as f32;

            // Calculate the reciprocal of the standard deviation (rstd)
            let s = 1.0 / (v + eps).sqrt();

            // Apply normalization, scale, and shift
            for i in 0..c {
                let n = (x[i] - m) * s; // normalized output
                let o = n * weight[i] + bias[i]; // scale and shift
                out_bt[i] = o; // write result to output
            }

            // Cache the mean and rstd for the backward pass later
            mean[b_idx * t + t_idx] = m;
            rstd[b_idx * t + t_idx] = s;
        }
    }
}

fn main() {
    // Define dimensions
    let b = 2; // Batch size
    let t = 3; // Sequence length
    let c = 4; // Number of features

    // Calculate total sizes
    let inp_size = b * t * c;
    let param_size = c;
    let stat_size = b * t;

    // Allocate memory
    let mut inp = vec![0.0f32; inp_size];
    let mut weight = vec![0.0f32; param_size];
    let mut bias = vec![0.0f32; param_size];
    let mut out = vec![0.0f32; inp_size];
    let mut mean = vec![0.0f32; stat_size];
    let mut rstd = vec![0.0f32; stat_size];

    // Initialize input tensor 'inp' with sequential values
    for i in 0..inp_size {
        inp[i] = (i + 1) as f32; // Fill with values 1.0, 2.0, 3.0, ...
    }

    // Initialize 'weight' to ones
    for i in 0..param_size {
        weight[i] = 1.0;
    }

    // Initialize 'bias' to zeros
    for i in 0..param_size {
        bias[i] = 0.0;
    }

    // Call the layernorm_forward function
    layernorm_forward(&mut out, &mut mean, &mut rstd, &inp, &weight, &bias, b, t, c);

    // Print the results
    println!("Input (inp):");
    for b_idx in 0..b {
        for t_idx in 0..t {
            print!("Batch {}, Time {}: ", b_idx, t_idx);
            for c_idx in 0..c {
                let idx = b_idx * t * c + t_idx * c + c_idx;
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

    println!("\nMean (mean):");
    for b_idx in 0..b {
        print!("Batch {}: ", b_idx);
        for t_idx in 0..t {
            let idx = b_idx * t + t_idx;
            print!("{:.4} ", mean[idx]);
        }
        println!();
    }

    println!("\nReciprocal Std Dev (rstd):");
    for b_idx in 0..b {
        print!("Batch {}: ", b_idx);
        for t_idx in 0..t {
            let idx = b_idx * t + t_idx;
            print!("{:.4} ", rstd[idx]);
        }
        println!();
    }
}

/* C Code

Mean (mean):
Batch 0: 2.5000 6.5000 10.5000
Batch 1: 14.5000 18.5000 22.5000

Reciprocal Std Dev (rstd):
Batch 0: 0.8944 0.8944 0.8944
Batch 1: 0.8944 0.8944 0.8944

Rust Code

Mean (mean):
Batch 0: 2.5000 6.5000 10.5000
Batch 1: 14.5000 18.5000 22.5000

Reciprocal Std Dev (rstd):
Batch 0: 0.8944 0.8944 0.8944
Batch 1: 0.8944 0.8944 0.8944
*/