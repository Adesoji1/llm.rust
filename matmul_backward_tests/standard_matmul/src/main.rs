use std::time::Instant;

use clap::Parser;
use serde_json::Value;


#[derive(Parser, Debug)]
#[command(name = "Input Params")]
struct Cli {
    #[arg(short, long)]
    b: usize,

    #[arg(short, long)]
    t: usize,

    #[arg(short, long)]
    c: usize,

    #[arg(short, long)]
    oc: usize,
}


fn matmul_backward_standard(
    dinp: &mut [f32],
    dweight: &mut [f32],
    mut dbias: Option<&mut [f32]>, // Declare dbias as mutable
    dout: &mut [f32],
    inp: &[f32],
    weight: &[f32],
    b: usize,
    t: usize,
    c: usize,
    oc: usize,
) {
    // Compute gradients into dinp
    for bb in 0..b {
        for tt in 0..t {
            let dout_offset = (bb * t + tt) * oc;
            let dinp_offset = (bb * t + tt) * c;
            for oo in 0..oc {
                let w_row_offset = oo * c;
                let d = dout[dout_offset + oo];

                for i in 0..c {
                    dinp[dinp_offset + i] += weight[w_row_offset + i] * d;
                }
            }
        }
    }

    // Compute gradients into dweight and dbias
    for oo in 0..oc {
        let dw_row_offset = oo * c;
        for bb in 0..b {
            for tt in 0..t {
                let dout_offset = (bb * t + tt) * oc + oo;
                let inp_offset = (bb * t + tt) * c;

                let d = dout[dout_offset];

                // Update dbias if it exists
                if let Some(dbias_vec) = dbias.as_mut() {
                    dbias_vec[oo] += d;
                }

                for i in 0..c {
                    dweight[dw_row_offset + i] += inp[inp_offset + i] * d;
                }
            }
        }
    }
}

fn matmul_forward_standard(
    out: &mut [f32],
    inp: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    b: usize,
    t: usize,
    c: usize,
    oc: usize,
) {

    for bb in 0..b {
        for tt in 0..t {
            let out_offset = (bb * t + tt) * oc;
            let inp_offset = (bb * t + tt) * c;
            let inp_bt = &inp[inp_offset..inp_offset + c];

            for oo in 0..oc {
                let mut val = if let Some(bias_vec) = bias {
                    bias_vec[oo]
                } else {
                    0.0
                };
                let weight_offset = oo * c;
                let w_row = &weight[weight_offset..weight_offset + c];

                for i in 0..c {
                    val += inp_bt[i] * w_row[i];
                }
                out[out_offset + oo] = val;
            }
        }
    }
}


fn main() {
    let cli = Cli::parse();
    let b = cli.b as usize;
    let t = cli.t as usize;
    let c = cli.c as usize;
    let oc = cli.oc as usize;
    let inp_size = b * t * c;     // Size of the input array
    let weight_size = c * oc;     // Size of the weight matrix
    let out_size = b * t * oc;    // Size of the output array
    println!("inp_size: {inp_size}, weight_size: {weight_size}, out_size: {out_size}");
    // Initialize the input array with sample data
    let inp: Vec<f32> = (0..inp_size)
        .map(|i| (i % 10) as f32) // Sample data: 0,1,2,...,9,0,1,2,...
        .collect();

    // Initialize the weight matrix with sample data
    let weight: Vec<f32> = (0..weight_size)
        .map(|i| ((i % 5) as f32) * 0.1 + 0.5) // Sample data: 0.5,0.6,0.7,...
        .collect();

    // Initialize the bias vector
    let bias = Some(vec![0.1_f32; oc]); // Bias of 0.1 for each output channel
    // Output tensor (B x T x OC)
    let mut out = vec![0.0f32; b*t*oc];
    // Perform the matrix multiplication
    let now = Instant::now();
    matmul_forward_standard(&mut out, &inp, &weight, bias.as_deref(),b,t,c,oc);
    let elapsed = now.elapsed();
    println!("Time taken: {:?}", elapsed);
    // do the backward pass
    let mut dinp = vec![0.0f32; b*t*c];
    let mut dweight = vec![0.0f32; c*oc];
    let mut dbias = vec![0.0f32; oc];
    let mut dout: Vec<f32> = (0..(b * t * oc))
    .map(|idx| (idx % 10) as f32 * 0.1)
    .collect();
    let now = Instant::now();
    matmul_backward_standard(&mut dinp, &mut dweight, Some(&mut dbias[..]), &mut dout, &inp, &weight, b, t, c, oc);
    let elapsed = now.elapsed();
    println!("Time taken: {:?}", elapsed);
    //println!("dout {:?}", dout);
    //println!("dinp {:?}", dinp);
    //println!("Output Tensor:");
    //println!("First batch {:?}", &out[0..10]);


}

/*
C CODE
dOutput Tensor:
Batch 0:
  Time step 0: 0.00 0.10 0.20 0.30 0.40
  Time step 1: 0.50 0.60 0.70 0.80 0.90
  Time step 2: 0.00 0.10 0.20 0.30 0.40
Batch 1:
  Time step 0: 0.50 0.60 0.70 0.80 0.90
  Time step 1: 0.00 0.10 0.20 0.30 0.40
  Time step 2: 0.50 0.60 0.70 0.80 0.90


Rust code
dOutput Tensor:
[0.0, 0.1, 0.2, 0.3, 0.4,
 0.5, 0.6, 0.7, 0.8, 0.90000004,
 0.0, 0.1, 0.2, 0.3, 0.4,

 0.5, 0.6, 0.7, 0.8, 0.90000004,
 0.0, 0.1, 0.2, 0.3, 0.4,
 0.5, 0.6, 0.7, 0.8, 0.90000004]
*/