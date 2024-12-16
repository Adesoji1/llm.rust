# llm.rust

llm.rust is my attempt to get more into Rust, after the inspiring work made by [Karpathy with llm.c](https://github.com/karpathy/llm.c). The code is pure simple Rust, no need for CUDA, and it runs well on CPUs - so no need of GPUs. The idea is to have a simple toolkit, written in Rust, that can run on any given GPT-like model's weights, so that we can continue the training steps and we can also fine tune the model given an input corpus.

There are many things to improve, for example:
- move the `Vec[T]` to `'a` lifetime for all the arrays we have in the code
- simplify the way we're slicing and copying arrays
- make the code more readable, so moving architecture out of the main training code.

## How to use this repo

If you want to run the code, you can test it against Karpathy's [small Shakespeare text `tinyshakespeare`](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt). You can download the text using the python code:
```bash
python prepro_tinyshakespeare.py
```
This will download the input corpus in a `data` folder. The text is convert the input text to input training and validation tokens (`tiny_shakespeare_train.bin` and `tiny_shakespeare_val.bin` respectively). The text is tokenised with GPT-2 tokenizer.

Then, you can build and run the Rust code on these data with:
```bash
cd llm
bash build.sh
```

This will run the training, along with validation tests and some inference. This is an example of output you should get from the code:
```bash
Number of parameters: 124439808
params_memory size: 124439808
Model Configuration:
Max Sequence Length: 1024
Vocabulary Size: 50257
Number of Layers: 12
Number of Heads: 12
Channels: 768
First 10 elements of params_memory:
params_memory[0] = -0.11010301
params_memory[1] = -0.039266724
params_memory[2] = 0.033107508
params_memory[3] = 0.13382645
params_memory[4] = -0.048475694
params_memory[5] = -0.078917675
params_memory[6] = -0.23977417
params_memory[7] = -0.08947388
params_memory[8] = 0.025254967
params_memory[9] = -0.107396826
Parameter sizes:
param_sizes[0] = 38597376
param_sizes[1] = 786432
param_sizes[2] = 9216
param_sizes[3] = 9216
param_sizes[4] = 21233664
param_sizes[5] = 27648
param_sizes[6] = 7077888
param_sizes[7] = 9216
param_sizes[8] = 9216
param_sizes[9] = 9216
param_sizes[10] = 28311552
param_sizes[11] = 36864
param_sizes[12] = 28311552
param_sizes[13] = 9216
param_sizes[14] = 768
param_sizes[15] = 768
Num batches: 1192
Step: 0
Time taken for forward pass: 704.785709ms
train loss: 5.356085
Time taken for backward pass: 540.8035ms
!!!! val loss: 4.828099
Step: 1
Time taken for forward pass: 450.922209ms
train loss: 4.300641
Time taken for backward pass: 472.333291ms
Step: 2
Time taken for forward pass: 251.624875ms
train loss: 4.6230845
Time taken for backward pass: 540.731792ms
Step: 3
Time taken for forward pass: 261.636417ms
train loss: 4.5993633
Time taken for backward pass: 473.1045ms
Step: 4
Time taken for forward pass: 265.662875ms
train loss: 4.6166663
Time taken for backward pass: 441.045916ms
Step: 5
Time taken for forward pass: 248.803666ms
train loss: 4.231429
Time taken for backward pass: 432.05875ms
Step: 6
Time taken for forward pass: 248.580792ms
train loss: 3.7531617
Time taken for backward pass: 435.134833ms
Step: 7
Time taken for forward pass: 248.81525ms
train loss: 3.6504595
Time taken for backward pass: 438.9085ms
Step: 8
Time taken for forward pass: 249.155458ms
train loss: 4.182244
Time taken for backward pass: 442.935458ms
Step: 9
Time taken for forward pass: 254.185084ms
train loss: 4.19958
Time taken for backward pass: 438.501ms
Step: 10
Time taken for forward pass: 249.539042ms
train loss: 4.2886634
Time taken for backward pass: 437.485708ms
Step: 11
Time taken for forward pass: 249.333334ms
train loss: 3.5606396
Time taken for backward pass: 439.324708ms
Step: 12
Time taken for forward pass: 249.053666ms
train loss: 3.7314389
Time taken for backward pass: 441.631625ms
Step: 13
Time taken for forward pass: 250.292625ms
train loss: 4.1585107
Time taken for backward pass: 441.02525ms
Step: 14
Time taken for forward pass: 250.06875ms
train loss: 3.8856323
Time taken for backward pass: 437.955417ms
Step: 15
Time taken for forward pass: 248.439667ms
train loss: 3.766488
Time taken for backward pass: 442.295666ms
Step: 16
Time taken for forward pass: 250.064958ms
train loss: 4.144007
Time taken for backward pass: 442.780209ms
Step: 17
Time taken for forward pass: 249.781292ms
train loss: 3.961168
Time taken for backward pass: 437.802416ms
Step: 18
Time taken for forward pass: 250.981375ms
train loss: 3.7960434
Time taken for backward pass: 444.495209ms
Step: 19
Time taken for forward pass: 248.624ms
train loss: 3.3710413
Time taken for backward pass: 442.205708ms
Step: 20
Time taken for forward pass: 249.413291ms
train loss: 3.8827891
Time taken for backward pass: 442.158083ms
!!!! val loss: 4.231359
Inference
Generated text:
```

Example of output (token, word):

```
3792, Is
340,  it
922,  good
11, ,
611,  if
345,  you
423,  have
26246,  pity
11, ,
284,  to
423,  have
281,  an
45618,  ornament
11, ,
257,  a
1486,  design
11, ,
198,

2514, To
9280,  dance
11, ,
7365,  bat
258, he
11, ,
18044,  breathe
290,  and
4545,  teach
30, ?
440,  O
11, ,
611,  if
340,  it
307,  be
2081,  true
11, ,
198,

1026, It
318,  is
2081,  true
356,  we
743,  may
307,  be
991,  still
2877,  living
11, ,
611,  if
340,  it
307,  be
2081,  true
25, :
198,

46, O
11, ,
2652,  stay
11, ,
393,  or
314,  I
2236,  shall
307,  be
2636,  dead
13, .
628,
```