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
    //println!("Output Tensor:");
    //println!("First batch {:?}", &out[0..10]);


}

// final output
/* out_offset 25, out [
0.54, 0.48000002, 0.47, 0.51, 0.6,
1.5799999, 1.56, 1.5900002, 1.67, 1.8000001,
1.12, 1.34, 1.6099999, 1.43, 1.3000001,
1.06, 1.02, 1.03, 1.09, 1.2,
2.1000001, 2.1, 2.15, 2.25, 2.4,
0.54, 0.48000002, 0.47, 0.51, 0.6]
*/

/* C output
Batch 0:
  Time step 0: 0.54 0.48 0.47 0.51 0.60
  Time step 1: 1.58 1.56 1.59 1.67 1.80
  Time step 2: 1.12 1.34 1.61 1.43 1.30
Batch 1:
  Time step 0: 1.06 1.02 1.03 1.09 1.20
  Time step 1: 2.10 2.10 2.15 2.25 2.40
  Time step 2: 0.54 0.48 0.47 0.51 0.60
   */