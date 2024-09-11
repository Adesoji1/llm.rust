use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::fs::File;
use std::path::Path;
use byteorder::{ReadBytesExt, LittleEndian};

const NUM_PARAMETER_TENSORS: usize = 16;
const NUM_ACTIVATION_TENSORS: usize = 23;


//***** UTILITY FUNCTION **** */
fn encoder_forward(out: &mut [f32], inp: &[i32], wte: &[f32], wpe: &[f32], b: usize, t: usize, c: usize) {
    println!("b: {}, t: {}, c: {}", b, t, c);
    for b_idx in 0..b {
        for t_idx in 0..t {
            let out_start_idx = b_idx * t * c + t_idx * c;
            let out_bt = &mut out[out_start_idx..out_start_idx + c];
            // Get the index of the token at inp[b, t]
            let ix = inp[b_idx * t + t_idx] as usize;  // Convert to usize for safe indexing
            let wte_start_idx = ix * c;
            let wte_ix = &wte[wte_start_idx..wte_start_idx + c];
            let wpe_start_idx = t_idx * c;
            let wpe_t = &wpe[wpe_start_idx..wpe_start_idx + c];
            // Add the two vectors and store the result in out[b, t, :]
            for i in 0..c {
                out_bt[i] = wte_ix[i] + wpe_t[i];
            }
        }
    }
}

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


fn matmul_forward(
    out: &mut [f32],
    inp: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>, // option because we may have None
    b: usize,
    t: usize,
    c: usize,
    oc: usize,
) {
    // Main multiplication function
    // OC is output channels
    // input is (B, T, C), weight is (OC, C), bias is (OC)
    // output will be (B, T, OC)


    for (b_idx, chunk) in inp.chunks(t * c).enumerate().take(b) {
        for (t_idx, inp_bt) in chunk.chunks(c).enumerate().take(t) {
            let out_bt = &mut out[b_idx * t * oc + t_idx * oc..][..oc];
            for (o, output) in out_bt.iter_mut().enumerate().take(oc) {
                let bias_val = bias.map_or(0.0, |b| b[o]);
                let weight_row = &weight[o * c..][..c];
                let val = inp_bt
                    .iter()
                    .zip(weight_row.iter())
                    .fold(bias_val, |acc, (&inp_val, &weight_val)| acc + inp_val * weight_val);
                *output = val;
            }
        }
    }
    /* Medium test:
    use rayon::prelude::*;

    inp.par_chunks(t * c).enumerate().take(b).for_each(|(b_idx, chunk)| {
        // Similar inner loops as above
    }); */
}


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
    println!("b: {}, t: {}, c: {}", b, t, c);
    println!("Size of preatt {}", preatt.len());
    println!("Size of att {}", att.len());
    println!("Size of inp {}", inp.len());
    for b_idx in 0..b {
        for t_idx in 0..t {
            for h in 0..nh {
                let query_start = b_idx * t * c3 + t_idx * c3 + h * hs;
                let preatt_start = b_idx * nh * t * t + h * t * t + t_idx * t;
                let att_start = b_idx * nh * t * t + h * t * t + t_idx * t;

                // Pass 1: calculate query dot key and maxval
                let mut maxval = f32::NEG_INFINITY;
                for t2 in 0..=t_idx {
                    let key_start = b_idx * t * c3 + t2 * c3 + h * hs + c; // +C because it's key

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
                    let expv = ((preatt[preatt_start + t2] - maxval).exp()).min(f32::MAX);
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
                    let value_start = b_idx * t * c3 + t2 * c3 + h * hs + c * 2; // +C*2 because it's value
                    let att_val = att[att_start + t2];
                    for i in 0..hs {
                        out[out_start + i] += att_val * inp[value_start + i];
                    }
                }
            }
        }
    }
}


fn residual_forward(out: &mut [f32], inp1: &[f32], inp2: &[f32], n: usize) {
    // Ensure that all slices have at least 'n' elements
    assert!(out.len() >= n && inp1.len() >= n && inp2.len() >= n, "Input slices must be of at least size n");

    for i in 0..n {
        out[i] = inp1[i] + inp2[i];
    }
    /* Iterator implementation
    fn residual_forward(out: &mut [f32], inp1: &[f32], inp2: &[f32]) {
    // Iterator-based approach, automatically handling bounds and potentially more idiomatic
    for ((o, i1), i2) in out.iter_mut().zip(inp1.iter()).zip(inp2.iter()) {
        *o = i1 + i2;
            }
    }
    */
}

fn gelu_forward(out: &mut [f32], inp: &[f32], n: usize) {
    let s = (2.0 / std::f32::consts::PI).sqrt();
    for i in 0..n {
        let x = inp[i];
        let cube: f32 = 0.044715 * x * x * x;
        out[i] = 0.5 * x * (1.0 + (s * (x + cube)).tanh());
    }
}

fn softmax_forward(out: &mut [f32], logits: &[f32], b: usize, t: usize, v: usize){
    // here Karpathy uses pragma
    for b_idx in 0..b{
        for t_idx in 0..t {
            let start_idx = b_idx * t * v + t_idx * v;
            let logits_bt = &logits[start_idx..start_idx + v];
            let out_bt = &mut out[start_idx..start_idx + v];
            let mut max_val = f32::NEG_INFINITY;
            for i in 0..v {
                if logits_bt[i] > max_val {
                    max_val = logits_bt[i];
                }
            }
            let mut sum = 0.0;
            for i in 0..v {
                let val = (logits_bt[i] - max_val).exp();
                out_bt[i] = val;
                sum += val;
            }
            let sum_inv = if sum == 0.0 { 0.0 } else { 1.0 / sum };
            for i in 0..v {
                out_bt[i] *= sum_inv;
            }
        }
    }
}

fn crossentropy_forward(out: &mut [f32], probs: &[f32], targets: &[i32], b: usize, t: usize, v: usize){
    for b_idx in 0..b {
        for t_idx in 0..t {
            let target = targets[b_idx * t + t_idx] as usize; // int ix
            let start_idx = b_idx * t * v + t_idx * v; // index
            let probs_bt = &probs[start_idx..start_idx + v]; //  probs_bt
            out[b_idx * t + t_idx] = -probs_bt[target].ln();
        }
    }
}

//******** DATALOADER CONFIGURATIONS ******//
struct DataLoader {
    B: usize,
    T: usize,
    tokens_file: BufReader<File>,
    file_size: u64,
    current_position: u64,
    batch: Vec<i32>,
    inputs: Vec<i32>,
    targets: Vec<i32>,
    num_batches: usize,
}

impl DataLoader {
    fn new(file_path: &Path, B: usize, T: usize) -> io::Result<Self>{
        let file = File::open(file_path)?;
        let mut reader = BufReader::new(file);
        let file_size = reader.seek(io::SeekFrom::End(0))?;
        reader.seek(io::SeekFrom::Start(0))?;
        // Good lesson to show, trait is not in scope, we need to import it
        if file_size < ((B*T+1)*std::mem::size_of::<i32>() as usize) as u64{
            return Err(io::Error::new(io::ErrorKind::Other, "File too small"));
        }
        let mut loader = DataLoader{
            B,
            T,
            tokens_file: reader,
            file_size,
            current_position: 0,
            batch: vec![0; B*T+1],
            inputs: vec![0; B*T+1],
            targets: vec![0; B*T+1],
            num_batches: (file_size / (B as u64*T as u64*std::mem::size_of::<i32>() as u64)) as usize,
        };
        loader.inputs = loader.batch[0..].to_vec();
        loader.targets = loader.batch[1..].to_vec();

        Ok(loader)
        }
    fn reset(&mut self) -> io::Result<()> {
        // I added this seek to Start 0
        //self.tokens_file.seek(SeekFrom::Start(0))?;
        // this is the original bit
        self.current_position = 0;
        Ok(())
    }
    fn next_batch(&mut self) -> io::Result<()>{
        if self.current_position + (self.B * self.T +1) as u64 * std::mem::size_of::<i32>() as u64 > self.file_size {
            self.current_position = 0;
        }
        self.tokens_file.seek(SeekFrom::Start(self.current_position))?;
        let buffer = self.batch.as_mut_slice();
        let bytes_to_read = buffer.len() * std::mem::size_of::<i32>();
        self.tokens_file.read_exact(bytemuck::cast_slice_mut(buffer))?; // bytemuck is a crate that provides safe and efficient byte conversion functions for Rust
        self.current_position += self.B as u64 * self.T as u64 *std::mem::size_of::<i32>() as u64;
        Ok(())
    }

}
/* END OF DATALOADER CONFIGURATION */

//****** GPT2 CONFIGURATIONS ********//
struct GPT2Config {
    max_seq_len: usize,
    vocab_size: usize,
    num_layers: usize,
    num_heads: usize,
    channels: usize,
}

struct ParameterTensors {
    wte: Vec<f32>, // (V, C)
    wpe: Vec<f32>, // (maxT, C)
    ln1w: Vec<f32>, // (L, C)
    ln1b: Vec<f32>, // (L, C)
    qkvw: Vec<f32>, // (L, 3*C, C)
    qkvb: Vec<f32>, // (L, 3*C)
    attprojw: Vec<f32>, // (L, C, C)
    attprojb: Vec<f32>, // (L, C)
    ln2w: Vec<f32>, // (L, C)
    ln2b: Vec<f32>, // (L, C)
    fcw: Vec<f32>, // (L, 4*C, C)
    fcb: Vec<f32>, // (L, 4*C)
    fcprojw: Vec<f32>, // (L, C, 4*C)
    fcprojb: Vec<f32>, // (L, C)
    lnfw: Vec<f32>, // (C)
    lnfb: Vec<f32>, // (C)
}

struct ActivationTensors {
    encoded: Vec<f32>, // (B, T, C)
    ln1: Vec<f32>, // (L, B, T, C)
    ln1_mean: Vec<f32>, // (L, B, T)
    ln1_rstd: Vec<f32>, // (L, B, T)
    qkv: Vec<f32>, // (L, B, T, 3*C)
    atty: Vec<f32>, // (L, B, T, C)
    preatt: Vec<f32>, // (L, B, NH, T, T)
    att: Vec<f32>, // (L, B, NH, T, T)
    attproj: Vec<f32>, // (L, B, T, C)
    residual2: Vec<f32>, // (L, B, T, C)
    ln2: Vec<f32>, // (L, B, T, C)
    ln2_mean: Vec<f32>, // (L, B, T)
    ln2_rstd: Vec<f32>, // (L, B, T)
    fch: Vec<f32>, // (L, B, T, 4*C)
    fch_gelu: Vec<f32>, // (L, B, T, 4*C)
    fcproj: Vec<f32>, // (L, B, T, C)
    residual3: Vec<f32>, // (L, B, T, C)
    lnf: Vec<f32>, // (B, T, C)
    lnf_mean: Vec<f32>, // (B, T)
    lnf_rstd: Vec<f32>, // (B, T)
    logits: Vec<f32>, // (B, T, V)
    probs: Vec<f32>, // (B, T, V)
    losses: Vec<f32>, // (B, T)
}

/*Since Rust doesn't have implicit nullability and raw pointers, we often use owned types like Vec<T> for dynamic arrays and manage explicit lifetimes where necessary.
*/
struct GPT2 {
    config: GPT2Config,
    params: ParameterTensors,
    param_sizes: Vec<usize>,
    params_memory: Vec<f32>,
    num_parameters: usize,
    grads: ParameterTensors,
    grads_memory: Vec<f32>,
    m_memory: Vec<f32>,
    v_memory: Vec<f32>,
    acts: ActivationTensors,
    act_sizes: Vec<usize>,
    acts_memory: Vec<f32>,
    num_activations: usize,
    grads_acts: ActivationTensors,
    grads_acts_memory: Vec<f32>,
    batch_size: usize,
    seq_len: usize,
    inputs: Vec<i32>, // Vector of integers
    targets: Vec<i32>, // Vector of integers
    mean_loss: f32,
}

impl GPT2 {
    fn new() -> Self {
        GPT2 {
            config: GPT2Config {
                max_seq_len: 0,
                vocab_size: 0,
                num_layers: 0,
                num_heads: 0,
                channels: 0,
            },
            params: ParameterTensors {
                wte: Vec::new(),
                wpe: Vec::new(),
                ln1w: Vec::new(),
                ln1b: Vec::new(),
                qkvw: Vec::new(),
                qkvb: Vec::new(),
                attprojw: Vec::new(),
                attprojb: Vec::new(),
                ln2w: Vec::new(),
                ln2b: Vec::new(),
                fcw: Vec::new(),
                fcb: Vec::new(),
                fcprojw: Vec::new(),
                fcprojb: Vec::new(),
                lnfw: Vec::new(),
                lnfb: Vec::new(),
            },
            param_sizes: vec![0; NUM_PARAMETER_TENSORS],
            params_memory: Vec::new(),
            num_parameters: 0,
            grads: ParameterTensors {
                wte: Vec::new(),
                wpe: Vec::new(),
                ln1w: Vec::new(),
                ln1b: Vec::new(),
                qkvw: Vec::new(),
                qkvb: Vec::new(),
                attprojw: Vec::new(),
                attprojb: Vec::new(),
                ln2w: Vec::new(),
                ln2b: Vec::new(),
                fcw: Vec::new(),
                fcb: Vec::new(),
                fcprojw: Vec::new(),
                fcprojb: Vec::new(),
                lnfw: Vec::new(),
                lnfb: Vec::new(),
            },
            grads_memory: Vec::new(),
            m_memory: Vec::new(),
            v_memory: Vec::new(),
            acts: ActivationTensors {
                encoded: Vec::new(),
                ln1: Vec::new(),
                ln1_mean: Vec::new(),
                ln1_rstd: Vec::new(),
                qkv: Vec::new(),
                atty: Vec::new(),
                preatt: Vec::new(),
                att: Vec::new(),
                attproj: Vec::new(),
                residual2: Vec::new(),
                ln2: Vec::new(),
                ln2_mean: Vec::new(),
                ln2_rstd: Vec::new(),
                fch: Vec::new(),
                fch_gelu: Vec::new(),
                fcproj: Vec::new(),
                residual3: Vec::new(),
                lnf: Vec::new(),
                lnf_mean: Vec::new(),
                lnf_rstd: Vec::new(),
                logits: Vec::new(),
                probs: Vec::new(),
                losses: Vec::new(),
            },
            act_sizes: vec![0; NUM_ACTIVATION_TENSORS],
            acts_memory: Vec::new(),
            num_activations: 0,
            grads_acts: ActivationTensors {
                encoded: Vec::new(),
                ln1: Vec::new(),
                ln1_mean: Vec::new(),
                ln1_rstd: Vec::new(),
                qkv: Vec::new(),
                atty: Vec::new(),
                preatt: Vec::new(),
                att: Vec::new(),
                attproj: Vec::new(),
                residual2: Vec::new(),
                ln2: Vec::new(),
                ln2_mean: Vec::new(),
                ln2_rstd: Vec::new(),
                fch: Vec::new(),
                fch_gelu: Vec::new(),
                fcproj: Vec::new(),
                residual3: Vec::new(),
                lnf: Vec::new(),
                lnf_mean: Vec::new(),
                lnf_rstd: Vec::new(),
                logits: Vec::new(),
                probs: Vec::new(),
                losses: Vec::new(),
            },
            grads_acts_memory: Vec::new(),
            batch_size: 0,
            seq_len: 0,
            inputs: Vec::new(),
            targets: Vec::new(),
            mean_loss: -1.0,
        }
    }
    /* UTILITY FUNCTION TO INITIALIZE ACTIVITY TENSOR */
    fn allocate_activation_tensors(&mut self, b: usize, t: usize, l: usize, nh: usize, c: usize, v: usize) {
        self.acts.encoded.resize(b * t * c, 0.0);
        self.acts.ln1.resize(l * b * t * c, 0.0);
        self.acts.ln1_mean.resize(l * b * t, 0.0);
        self.acts.ln1_rstd.resize(l * b * t, 0.0);
        self.acts.qkv.resize(l * b * t * 3 * c, 0.0);
        self.acts.atty.resize(l * b * t * c, 0.0);
        self.acts.preatt.resize(l * b * nh * t * t, 0.0);
        self.acts.att.resize(l * b * nh * t * t, 0.0);
        self.acts.attproj.resize(l * b * t * c, 0.0);
        self.acts.residual2.resize(l * b * t * c, 0.0);
        self.acts.ln2.resize(l * b * t * c, 0.0);
        self.acts.ln2_mean.resize(l * b * t, 0.0);
        self.acts.ln2_rstd.resize(l * b * t, 0.0);
        self.acts.fch.resize(l * b * t * 4 * c, 0.0);
        self.acts.fch_gelu.resize(l * b * t * 4 * c, 0.0);
        self.acts.fcproj.resize(l * b * t * c, 0.0);
        self.acts.residual3.resize(l * b * t * c, 0.0);
        self.acts.lnf.resize(b * t * c, 0.0);
        self.acts.lnf_mean.resize(b * t, 0.0);
        self.acts.lnf_rstd.resize(b * t, 0.0);
        self.acts.logits.resize(b * t * v, 0.0);
        self.acts.probs.resize(b * t * v, 0.0);
        self.acts.losses.resize(b * t, 0.0);


    }
    /* FORWARD PASS */
    pub fn forward(&mut self, inputs: &[i32], targets: Option<&[i32]>, b: usize, t: usize) -> io::Result<()> {
        // Ensure the model is properly initialized
        if self.params_memory.is_empty() {
            return Err(io::Error::new(io::ErrorKind::Other, "Error: model was not initialized properly."));
        }

        let v = self.config.vocab_size;
        let l = self.config.num_layers;
        let nh = self.config.num_heads;
        let c = self.config.channels;
        // allocate space for all the activations if needed
        if self.acts_memory.is_empty() {
            self.batch_size = b;
            self.seq_len = t;
            // Resize activation tensors based on the current configuration and batch settings
            self.allocate_activation_tensors(b, t, l, nh, c, v);
        } else {
            // Ensure B and T are not larger than what was previously allocated
            if b > self.batch_size || t > self.seq_len {
                return Err(io::Error::new(io::ErrorKind::InvalidInput, "Batch size or sequence length is too large."));
            }
        }

        // Cache the inputs and optionally the targets
        self.inputs = inputs.to_vec();
        println!("inputs size: {}", self.inputs.len());

        if let Some(targets) = targets {
            self.targets = targets.to_vec();
        }

        // Call encoder_forward
        //let out = vec![0.0; b * t * c]; // Output tensor for the encoder
        let wte = &self.params.wte;
        let wpe = &self.params.wpe;
        // print size of wte and wpe
        encoder_forward(&mut self.acts.encoded, &inputs, &wte, &wpe, b, t, c);
        // Process each layer
        for l in 0..self.config.num_layers {
            // Get the residual from the previous layer
            let index_base = l * self.batch_size * self.seq_len * self.config.channels; // L*B*T*C

            let next_index_base = (l + 1) * self.batch_size * self.seq_len * self.config.channels; // (L+1)*B*T*C

            let mut residual: Vec<f32> = if l == 0 {
                self.acts.encoded.clone()
            } else {
                self.acts.residual3[(index_base - 1)*self.batch_size * self.seq_len * self.config.channels..index_base].to_vec()
            };

            // Access layer-specific parameters
            /* Lesson to write on Medium
            In C we perform the pointer arithmetic
            performs pointer arithmetic to obtain the address of a segment within the ln1w array. This effectively moves the pointer l * C positions forward from the start of the array, which corresponds to the start of the weight matrix for the l-th layer.

            In Rust, direct pointer manipulation like this is generally avoided to maintain safety. Instead, Rust uses slices, which are safer because they maintain bounds-checking and other safety properties. When you write:

            */
            let l_ln1w = &self.params.ln1w[l * self.config.channels..(l + 1) * self.config.channels];
            let l_ln1b = &self.params.ln1b[l * self.config.channels..(l + 1) * self.config.channels];
            let l_qkvw = &self.params.qkvw[l * 3 * self.config.channels * self.config.channels..(l + 1) * 3 * self.config.channels * self.config.channels];
            let l_qkvb = &self.params.qkvb[l * 3 * self.config.channels..(l + 1) * 3 * self.config.channels];
            let l_attprojw = &self.params.attprojw[l * self.config.channels * self.config.channels..(l + 1) * self.config.channels * self.config.channels];
            let l_attprojb = &self.params.attprojb[l * self.config.channels..(l + 1) * self.config.channels];
            let l_ln2w = &self.params.ln2w[l * self.config.channels..(l + 1) * self.config.channels];
            let l_ln2b = &self.params.ln2b[l * self.config.channels..(l + 1) * self.config.channels];
            let l_fcw = &self.params.fcw[l * 4 * self.config.channels * self.config.channels..(l + 1) * 4 * self.config.channels * self.config.channels];
            let l_fcb = &self.params.fcb[l * 4 * self.config.channels..(l + 1) * 4 * self.config.channels];
            let l_fcprojw = &self.params.fcprojw[l * self.config.channels * 4 * self.config.channels..(l + 1) * self.config.channels * 4 * self.config.channels];
            let l_fcprojb = &self.params.fcprojb[l * self.config.channels..(l + 1) * self.config.channels];

            let base_idx = l * self.batch_size * self.seq_len;
            let c = self.config.channels;
            let nh = self.config.num_heads;

            // Activation slices for this layer
            let l_ln1 = &mut self.acts.ln1[base_idx * c..(base_idx + self.batch_size * self.seq_len) * c];
            let l_ln1_mean = &mut self.acts.ln1_mean[base_idx..base_idx + self.batch_size * self.seq_len];
            let l_ln1_rstd = &mut self.acts.ln1_rstd[base_idx..base_idx + self.batch_size * self.seq_len];
            let l_qkv = &mut self.acts.qkv[base_idx * 3 * c..(base_idx + self.batch_size * self.seq_len) * 3 * c];
            let l_atty = &mut self.acts.atty[base_idx * c..(base_idx + self.batch_size * self.seq_len) * c];
            let l_preatt = &mut self.acts.preatt[base_idx * nh * self.seq_len..(base_idx + self.batch_size * nh * self.seq_len) * self.seq_len];
            let l_att = &mut self.acts.att[base_idx * nh * self.seq_len..(base_idx + self.batch_size * nh * self.seq_len) * self.seq_len];
            let l_attproj = &mut self.acts.attproj[base_idx * c..(base_idx + self.batch_size * self.seq_len) * c];
            let l_residual2 = &mut self.acts.residual2[base_idx * c..(base_idx + self.batch_size * self.seq_len) * c];
            let l_ln2 = &mut self.acts.ln2[base_idx * c..(base_idx + self.batch_size * self.seq_len) * c];
            let l_ln2_mean = &mut self.acts.ln2_mean[base_idx..base_idx + self.batch_size * self.seq_len];
            let l_ln2_rstd = &mut self.acts.ln2_rstd[base_idx..base_idx + self.batch_size * self.seq_len];
            let l_fch = &mut self.acts.fch[base_idx * 4 * c..(base_idx + self.batch_size * self.seq_len) * 4 * c];
            let l_fch_gelu = &mut self.acts.fch_gelu[base_idx * 4 * c..(base_idx + self.batch_size * self.seq_len) * 4 * c];
            let l_fcproj = &mut self.acts.fcproj[base_idx * c..(base_idx + self.batch_size * self.seq_len) * c];
            let l_residual3 = &mut self.acts.residual3[base_idx * c..(base_idx + self.batch_size * self.seq_len) * c];

            // FORWARD PASS
            println!("Executing layernorm foward pass");
            layernorm_forward(
                 l_ln1,
                 l_ln1_mean,
                 l_ln1_rstd,
                & mut residual,
                &l_ln1w,  // weight for layernorm
                &l_ln1b,  // bias for layernorm
                self.batch_size,
                self.seq_len,
                self.config.channels
            );
            println!("Executing matmul forward pass");
            matmul_forward(
                l_qkv,
                l_ln1,      // Input
                l_qkvw,     // Weights
                Some(l_qkvb),     // Bias
                b,
                t,
                c,
                3*c
            );
            println!("Executing attention forward pass");
            attention_forward(
                l_atty,
                l_preatt,
                l_att,
                l_qkv,
                b,
                t,
                c,
                nh);
            println!("Executing matmul forward pass");
            matmul_forward(
                l_attproj,
                l_atty,
                l_attprojw,
                Some(l_attprojb),
                b,
                t,
                c,
                c);
            println!("Executing residual forward pass");
            residual_forward(
                l_residual2,
                &residual,
                l_attproj,
                b*t*c);
            println!("Executing layernorm forward pass");
            layernorm_forward(
                l_ln2,
                l_ln2_mean,
                l_ln2_rstd,
                l_residual2,
                l_ln2w,
                l_ln2b,
                b,
                t,
                c);
            println!("Executing matmul forward pass");
            matmul_forward(
                l_fch,
                l_ln2,
                l_fcw,
                Some(l_fcb),
                b,
                t,
                4*c,
                c);
            println!("Executing gelu forward pass");
            gelu_forward(
                l_fch_gelu,
                l_fch,
                b*t*4*c);
            println!("Executing matmul forward pass");
            matmul_forward(
                l_fcproj,
                l_fch_gelu,
                l_fcprojw,
                Some(l_fcprojb),
                b,
                t,
                4*c,
                c);
            println!("Executing residual forward pass");
            residual_forward(
                l_residual3,
                l_ln2,
                l_fcproj,
                b*t*c);
        }
        // line 758 of c code
        let last_layer_index = (l - 1) * b * t * c;
        let residual = &mut self.acts.residual3[last_layer_index..];
        layernorm_forward(
            &mut self.acts.lnf,
            &mut self.acts.lnf_mean,
            &mut self.acts.lnf_rstd,
            residual,
            &self.params.lnfw,
            &self.params.lnfb,
            b,
            t,
            c);
        matmul_forward(&mut self.acts.logits,
            &mut self.acts.lnf,
            & self.params.wte,
            None,
            b,
            t,
            c,
            v);
        softmax_forward(&mut self.acts.probs,
            &self.acts.logits,
            b,
            t,
            v);
        // line 764
        if let Some(targets) = targets {
            crossentropy_forward(&mut self.acts.losses, &self.acts.probs, targets, b, t, v);
            let mut loss = 0.0;
            for i in 0..b*t {
                loss += self.acts.losses[i];
            }
            self.mean_loss = loss / (b * t) as f32;
        }else{
            self.mean_loss = -1.0;
        }
        Ok(())
    }

}
/* END OF GPT2 CONFIGURATION */

fn gpt2_build_from_checkpoint(model: &mut GPT2, checkpoint_path: &Path) -> io::Result<()> {
    // Open the model file
    let mut file = BufReader::new(File::open(checkpoint_path)?);

    // Read in the model header
    let mut model_header = [0i32; 256];
    for i in 0..256 {
        model_header[i] = file.read_i32::<LittleEndian>()?;
    }

    if model_header[0] != 20240326 {
        return Err(io::Error::new(io::ErrorKind::Other, "Bad magic model file"));
    }
    if model_header[1] != 1 {
        return Err(io::Error::new(io::ErrorKind::Other, "Bad version in model file"));
    }

    // Read in hyperparameters
    let (max_t, v, l, nh, c) = (
        model_header[2] as usize,
        model_header[3] as usize,
        model_header[4] as usize,
        model_header[5] as usize,
        model_header[6] as usize,
    );

    // Setting the hyperparameters
    model.config = GPT2Config {
        max_seq_len: max_t,
        vocab_size: v,
        num_layers: l,
        num_heads: nh,
        channels: c,
    };

    // Calculate and store parameter sizes
    model.param_sizes = vec![
        v * c,
        max_t * c,
        l * c,
        l * c,
        l * (3 * c) * c,
        l * (3 * c),
        l * c * c,
        l * c,
        l * c,
        l * c,
        l * (4 * c) * c,
        l * (4 * c),
        l * c * (4 * c),
        l * c,
        c,
        c,
    ];

    let num_parameters: usize = model.param_sizes.iter().sum();
    println!{"Number of parameters: {}", num_parameters};
    model.num_parameters = num_parameters;

    // Allocate space for all parameters and read them in
    model.params_memory = vec![0.0; num_parameters];
    println!("params_memory size: {}", model.params_memory.len());
    for i in 0..num_parameters {
        model.params_memory[i] = file.read_f32::<LittleEndian>()?;
    }
    // littleendian: functionality for reading and writing numbers in either little-endian or big-endian byte order directly to and from byte arrays

    // read all teh input model params ugly implementation
    let mut offset = 0;
    model.params.wte = model.params_memory[offset..offset + model.param_sizes[0]].to_vec(); offset += model.param_sizes[0];
    model.params.wpe = model.params_memory[offset..offset + model.param_sizes[1]].to_vec(); offset += model.param_sizes[1];
    model.params.ln1w = model.params_memory[offset..offset + model.param_sizes[2]].to_vec(); offset += model.param_sizes[2];
    model.params.ln1b = model.params_memory[offset..offset + model.param_sizes[3]].to_vec(); offset += model.param_sizes[3];
    model.params.qkvw = model.params_memory[offset..offset + model.param_sizes[4]].to_vec(); offset += model.param_sizes[4];
    model.params.qkvb = model.params_memory[offset..offset + model.param_sizes[5]].to_vec(); offset += model.param_sizes[5];
    model.params.attprojw = model.params_memory[offset..offset + model.param_sizes[6]].to_vec(); offset += model.param_sizes[6];
    model.params.attprojb = model.params_memory[offset..offset + model.param_sizes[7]].to_vec(); offset += model.param_sizes[7];
    model.params.ln2w = model.params_memory[offset..offset + model.param_sizes[8]].to_vec(); offset += model.param_sizes[8];
    model.params.ln2b = model.params_memory[offset..offset + model.param_sizes[9]].to_vec(); offset += model.param_sizes[9];
    model.params.fcw = model.params_memory[offset..offset + model.param_sizes[10]].to_vec(); offset += model.param_sizes[10];
    model.params.fcb = model.params_memory[offset..offset + model.param_sizes[11]].to_vec(); offset += model.param_sizes[11];
    model.params.fcprojw = model.params_memory[offset..offset + model.param_sizes[12]].to_vec(); offset += model.param_sizes[12];
    model.params.fcprojb = model.params_memory[offset..offset + model.param_sizes[13]].to_vec(); offset += model.param_sizes[13];
    model.params.lnfw = model.params_memory[offset..offset + model.param_sizes[14]].to_vec(); offset += model.param_sizes[14];
    model.params.lnfb = model.params_memory[offset..offset + model.param_sizes[15]].to_vec(); offset += model.param_sizes[15];

    // Initialize other fields to defaults
    model.acts_memory = Vec::new();
    model.grads_memory = Vec::new();
    model.m_memory = Vec::new();
    model.v_memory = Vec::new();
    model.grads_acts_memory = Vec::new();
    model.inputs = Vec::new();
    model.targets = Vec::new();
    model.batch_size = 0;
    model.seq_len = 0;
    model.mean_loss = -1.0; // Indicate no loss calculated yet

    Ok(())
}

fn print_model_summary(model: &GPT2) {
    println!("Model Configuration:");
    println!("Max Sequence Length: {}", model.config.max_seq_len);
    println!("Vocabulary Size: {}", model.config.vocab_size);
    println!("Number of Layers: {}", model.config.num_layers);
    println!("Number of Heads: {}", model.config.num_heads);
    println!("Channels: {}", model.config.channels);

    // Print first few elements of params_memory
    println!("First 10 elements of params_memory:");
    model.params_memory.iter().take(10).enumerate().for_each(|(index, &value)| {
        println!("params_memory[{}] = {}", index, value);
    });

    // Print parameter sizes
    println!("Parameter sizes:");
    model.param_sizes.iter().enumerate().for_each(|(index, &size)| {
        println!("param_sizes[{}] = {}", index, size);
    });

    // If you have other vectors or arrays, you can add similar print statements here
}
fn main() {

    let mut model = GPT2::new();
    let checkpoint_path = Path::new("/Users/stefano.bosisio/Documents/llm.rust/gpt2_124M.bin");
    let _ = gpt2_build_from_checkpoint(&mut model,  &checkpoint_path);
    print_model_summary(&model);

    // debugging
    //print_model_summary(&model);
    let tiny_shakespeare_train: &Path = Path::new("/Users/stefano.bosisio/Documents/llm.rust/data/tiny_shakespeare_train.bin");
    let tiny_shakespeare_val: &Path = Path::new("/Users/stefano.bosisio/Documents/llm.rust/data/tiny_shakespeare_val.bin");
    // initialise B & T
    let B: usize = 4;
    let T: usize = 64;
    let val_num_batches = 10;
    // train loader
    let mut train_loader: DataLoader = DataLoader::new(tiny_shakespeare_train, B, T).unwrap();
    // debug print
    println!("Num batches: {}", train_loader.num_batches);
    // val loader
    let mut val_loader: DataLoader = DataLoader::new(tiny_shakespeare_val, B, T).unwrap();

    // training variables
    //let rng_state = 1337;
    const GEN_MAX_LENGTH: usize = 64; // move the const above
    //let mut gen_tokens = [0; GEN_MAX_LENGTH];
    // init of the model
    model.mean_loss = 0.0;
    for step in 0..40{
        // Once in a while estimate the validation loss
        if step % 10 == 0 {
            let mut val_loss = 0.0;
            val_loader.reset();
            for _ in 0..val_num_batches {
                val_loader.next_batch();
                model.forward(&val_loader.inputs, Some(&val_loader.targets), B, T);
                val_loss += model.mean_loss;
            }
            val_loss /= val_num_batches as f32;
            println!("val loss: {}", val_loss);
        }
        // Training step
        train_loader.reset();
        for _ in 0..train_loader.num_batches {
            train_loader.next_batch();
            model.forward(&train_loader.inputs, Some(&train_loader.targets), B, T);
            //print!("train loss: {}", model.mean_loss);
        }
    }
}