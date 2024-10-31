use rayon::prelude::*;
use std::time::Instant;
use cblas::{sgemm, Layout, Transpose};
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

fn matmul_blas(
    out: &mut [f32],
    inp: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    b: usize,   // Batch size
    t: usize,   // Time steps or sequence length
    c: usize,   // Input channels
    oc: usize,  // Output channels
) {
    let m = (b * t) as i32; // Number of rows of the output matrix
    let k = c as i32;       // Number of columns of the input matrix / rows of the weight matrix
    let n = oc as i32;      // Number of columns of the output matrix

    // Leading dimensions for Row-Major layout
    let lda = k; // lda >= K
    let ldb = k; // ldb >= N
    let ldc = n; // ldc >= N

    //println!("m: {m}, k: {k}, n: {n}, lda: {lda}, ldb: {ldb}, ldc: {ldc}");

    // Perform the matrix multiplication using BLAS sgemm
    unsafe {
        sgemm(
            Layout::RowMajor,
            Transpose::None, // Transpose of A ('N' for no transpose)
            Transpose::Ordinary, // Transpose of B
            m,
            n,
            k,
            1.0,
            inp,
            lda,
            weight,
            ldb,
            0.0,
            out,
            ldc,
        );
    }

    // Add bias if present
    if let Some(bias) = bias {
        out.par_chunks_mut(oc)
            .for_each(|row| {
                for (o, val) in row.iter_mut().enumerate() {
                    *val += bias[o];
                }
            });
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
    matmul_blas(&mut out, &inp, &weight, bias.as_deref(),b,t,c,oc);
    let elapsed = now.elapsed();
    println!("Time taken: {:?}", elapsed);
    //println!("out {:?}", out);
    //println!("Output Tensor:");
    //println!("First batch {:?}", &out[0..10]);


}

// final output
// The version of blas must be 0.9
// to work with clap
/* out_offset 25, out
out [0.54, 0.48000002, 0.47, 0.51, 0.6,
1.58, 1.5600001, 1.59, 1.67, 1.8000001,
1.1200001, 1.34, 1.61, 1.43, 1.3000001,
1.0600001, 1.02, 1.03, 1.09, 1.2
2.1, 2.1, 2.15, 2.25, 2.3999999,
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

According to the BLAS documentation for sgemm in Row-Major layout:

If Transpose of B is Transpose::None:

B has dimensions (k x n)
ldb should be n
If Transpose of B is Transpose::Ordinary:

B has dimensions (n x k) before transpose and (k x n) after transpose
ldb should be k
Explanation:

ldb specifies the leading dimension of B, which is the number of elements between successive rows (in memory).
When B is transposed, its leading dimension changes accordingly.


Implementation for testing with C


    let b = 2;    // Batch size
    let t = 3;    // Time steps or sequence length
    let c = 4;    // Input channels
    let oc = 5;   // Output channels
    // let b = 8; 64    // Batch size
    // let t = 32;  1024  // Time steps or sequence length
    // let c = 16;   768 // Input channels
    // let oc = 16;   768 // Output channels
    // Compute the sizes of the arrays

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

    /* Custom inputs */
    let inp: Vec<f32> = vec![
        // Batch 0
        0.0, 0.1, 0.2, 0.3,
        0.4, 0.5, 0.6, 0.7,
        0.8, 0.9, 0.0, 0.1,
        // Batch 1
        0.20, 0.30, 0.40, 0.50,
        0.60, 0.70, 0.80, 0.90,
        0.00, 0.10, 0.20, 0.30
    ];

    let weight: Vec<f32> = vec![
        // Output channel 0
        0.5, 0.6, 0.7, 0.8,
        // Output channel 1
        0.9, 0.5, 0.6, 0.7,
        // Output channel 2
        0.8, 0.9, 0.5, 0.6,
        // Output channel 3
        0.7, 0.8, 0.9, 0.5,
        // Output channel 4
        0.6, 0.7, 0.8, 0.9
    ];

    let bias = Some(vec![0.1_f32; oc]); // Bias of 0.1 for each output channel

    println!("inp {:?}", inp);
    println!("weight {:?}", weight);
    println!("bias {:?}", bias);
    println!("out {:?}", out);
*/