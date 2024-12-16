use std::time::Instant;
use rayon::prelude::*;
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


fn matmul_backward_rayon(
    dinp: &mut [f32],
    dweight: &mut [f32],
    dbias: Option<&mut [f32]>,
    dout: &[f32],
    inp: &[f32],
    weight: &[f32],
    B: usize,
    T: usize,
    C: usize,
    OC: usize,
) {
    use rayon::prelude::*;

    // Compute gradients into dinp
    dinp.par_chunks_mut(T * C)
        .enumerate()
        .for_each(|(bb, dinp_b)| {
            let dout_b = &dout[bb * T * OC..(bb + 1) * T * OC];
            for tt in 0..T {
                let dinp_bt = &mut dinp_b[tt * C..(tt + 1) * C];
                let dout_bt = &dout_b[tt * OC..(tt + 1) * OC];

                for oo in 0..OC {
                    let w_row = &weight[oo * C..(oo + 1) * C];
                    let d = dout_bt[oo];

                    for i in 0..C {
                        dinp_bt[i] += w_row[i] * d;
                    }
                }
            }
        });

    // Compute gradients into dweight and dbias
    match dbias {
        Some(dbias_vec) => {
            dweight.par_chunks_mut(C)
                .zip(dbias_vec.par_iter_mut())
                .enumerate()
                .for_each(|(oo, (dw_row, dbias_elem))| {
                    let mut dbias_local = 0.0f32;

                    for bb in 0..B {
                        for tt in 0..T {
                            let dout_offset = (bb * T + tt) * OC + oo;
                            let inp_offset = (bb * T + tt) * C;

                            let d = dout[dout_offset];

                            // Accumulate local dbias
                            dbias_local += d;

                            for i in 0..C {
                                dw_row[i] += inp[inp_offset + i] * d;
                            }
                        }
                    }

                    // Update dbias
                    *dbias_elem += dbias_local;
                });
        },
        None => {
            dweight.par_chunks_mut(C)
                .enumerate()
                .for_each(|(oo, dw_row)| {
                    for bb in 0..B {
                        for tt in 0..T {
                            let dout_offset = (bb * T + tt) * OC + oo;
                            let inp_offset = (bb * T + tt) * C;

                            let d = dout[dout_offset];

                            for i in 0..C {
                                dw_row[i] += inp[inp_offset + i] * d;
                            }
                        }
                    }
                });
        }
    }
}


fn matmul_forward_rayon(
    out: &mut [f32],
    inp: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    B: usize,
    T: usize,
    C: usize,
    OC: usize,
) {
    out.par_chunks_mut(T * OC)
        .zip(inp.par_chunks(T * C))
        .for_each(|(out_b, inp_b)| {
            for time_idx in 0..T {
                let inp_bt = &inp_b[time_idx * C..(time_idx + 1) * C];
                let out_bt = &mut out_b[time_idx * OC..(time_idx + 1) * OC];

                for o in 0..OC {
                    let mut val = bias.map_or(0.0, |b| b[o]);
                    let w_row = &weight[o * C..(o + 1) * C];
                    for i in 0..C {
                        val += inp_bt[i] * w_row[i];
                    }
                    out_bt[o] = val;
                }
            }
        });
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
    matmul_forward_rayon(&mut out, &inp, &weight, bias.as_deref(),b,t,c,oc);
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
    matmul_backward_rayon(
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
    println!("Time taken (backward with Rayon): {:?}", elapsed);
    //println!("Output Tensor:");
    //println!("{:?}", &dout);


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

Rust
[0.0, 0.1, 0.2, 0.3, 0.4,
 0.5, 0.6, 0.7, 0.8, 0.90000004,
 0.0, 0.1, 0.2, 0.3, 0.4,

 0.5, 0.6, 0.7, 0.8, 0.90000004,
 0.0, 0.1, 0.2, 0.3, 0.4,
 0.5, 0.6, 0.7, 0.8, 0.90000004]
   */