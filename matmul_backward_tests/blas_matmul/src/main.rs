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

fn matmul_backward_blas(
    dinp: &mut [f32],
    dweight: &mut [f32],
    dbias: Option<&mut [f32]>,
    dout: &[f32],
    inp: &[f32],
    weight: &[f32],
    b: usize,   // Batch size
    t: usize,   // Time steps or sequence length
    c: usize,   // Input channels
    oc: usize,  // Output channels
) {
    use cblas::{sgemm, Layout, Transpose};

    let m = (b * t) as i32; // Number of rows in dout and dinp
    let k = oc as i32;      // Number of columns in dout and rows in weight
    let n = c as i32;       // Number of columns in weight and dinp

    // Compute dinp = dout * weight
    unsafe {
        sgemm(
            Layout::RowMajor,
            Transpose::None,     // No transpose for dout
            Transpose::None,     // No transpose for weight
            m,
            n,
            k,
            1.0,
            dout,
            k,                   // lda >= K (set to k)
            weight,
            n,                   // ldb >= N (set to n)
            0.0,
            dinp,
            n,                   // ldc >= N (set to n)
        );
    }

    let m_dw = oc as i32;      // Number of rows in dweight and dout^T
    let k_dw = (b * t) as i32; // Number of columns in dout^T and rows in inp
    let n_dw = c as i32;       // Number of columns in inp and dweight

    // Compute dweight = dout^T * inp
    unsafe {
        sgemm(
            Layout::RowMajor,
            Transpose::Ordinary, // Transpose dout
            Transpose::None,     // No transpose for inp
            m_dw,
            n_dw,
            k_dw,
            1.0,
            dout,
            m_dw,                // lda >= M (set to m_dw)
            inp,
            n_dw,                // ldb >= N (set to n_dw)
            0.0,
            dweight,
            n_dw,                // ldc >= N (set to n_dw)
        );
    }

    // Compute dbias = sum over batches and time steps of dout
    if let Some(dbias) = dbias {
        for o in 0..oc {
            let mut sum = 0.0f32;
            for bt in 0..(b * t) {
                sum += dout[bt * oc + o];
            }
            dbias[o] += sum;
        }
    }
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
    // do the backward pass
    let mut dinp = vec![0.0f32; b*t*c];
    let mut dweight = vec![0.0f32; c*oc];
    let mut dbias = vec![0.0f32; oc];
    // Initialize dout tensor (B x T x OC)
    let dout: Vec<f32> = (0..(b * t * oc))
    .map(|idx| (idx % 10) as f32 * 0.1)
    .collect();
    let now = Instant::now();
    matmul_backward_blas(
        &mut dinp,
        &mut dweight,
        Some(&mut dbias[..]),
        &dout,
        &inp,
        &weight,
        b,
        t,
        c,
        oc,
    );
    let elapsed = now.elapsed();
    println!("Time taken: {:?}", elapsed);
    //println!("dout {:?}", &dout);


}


/* C output
dOutput Tensor:
Batch 0:
  Time step 0: 0.00 0.10 0.20 0.30 0.40
  Time step 1: 0.50 0.60 0.70 0.80 0.90
  Time step 2: 0.00 0.10 0.20 0.30 0.40
Batch 1:
  Time step 0: 0.50 0.60 0.70 0.80 0.90
  Time step 1: 0.00 0.10 0.20 0.30 0.40
  Time step 2: 0.50 0.60 0.70 0.80 0.90

Rust output
dout
[0.0, 0.1, 0.2, 0.3, 0.4,
 0.5, 0.6, 0.7, 0.8, 0.90000004,
 0.0, 0.1, 0.2, 0.3, 0.4,

 0.5, 0.6, 0.7, 0.8, 0.90000004,
 0.0, 0.1, 0.2, 0.3, 0.4,
 0.5, 0.6, 0.7, 0.8, 0.90000004]

*/