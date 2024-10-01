use cblas::{sgemm, Layout, Transpose};
use rayon::prelude::*;

fn matmul_forward(
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
    let ldb = n; // ldb >= N
    let ldc = n; // ldc >= N

    println!("m: {m}, k: {k}, n: {n}, lda: {lda}, ldb: {ldb}, ldc: {ldc}");

    // Perform the matrix multiplication using BLAS sgemm
    unsafe {
        sgemm(
            Layout::RowMajor,
            Transpose::None, // Transpose of A ('N' for no transpose)
            Transpose::None, // Transpose of B
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
       // Define dimensions
       let b = 2;    // Batch size
       let t = 3;    // Time steps or sequence length
       let c = 4;    // Input channels
       let oc = 5;   // Output channels
       println!("b: {b}, t: {t}, c: {c}, oc: {oc}");
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

       // Allocate the output array
       let mut out = vec![0.0_f32; out_size];

       // Call the matmul_forward function
       matmul_forward(
           &mut out,
           &inp,
           &weight,
           bias.as_deref(), // Convert Option<Vec<f32>> to Option<&[f32]>
           b,
           t,
           c,
           oc,
       );

       // Print the output
       println!("Output:");
       for (i, chunk) in out.chunks(oc).enumerate() {
           println!("Row {}: {:?}", i, chunk);
       }
}