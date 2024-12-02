#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// The attention_forward function as provided
void attention_forward(float* out, float* preatt, float* att,
                       float* inp,
                       int B, int T, int C, int NH) {
    // input is (B, T, 3C) Q,K,V
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    int C3 = C*3;
    int hs = C / NH; // head size
    float scale = 1.0f / sqrtf((float)hs);

    #pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                float* query_t = inp + b * T * C3 + t * C3 + h * hs;
                float* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
                float* att_bth = att + b*NH*T*T + h*T*T + t*T;

                // pass 1: calculate query dot key and maxval
                float maxval = -10000.0f; // TODO something better
                for (int t2 = 0; t2 <= t; t2++) {
                    float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

                    // (query_t) dot (key_t2)
                    float val = 0.0f;
                    for (int i = 0; i < hs; i++) {
                        val += query_t[i] * key_t2[i];
                    }
                    val *= scale;
                    if (val > maxval) {
                        maxval = val;
                    }

                    preatt_bth[t2] = val;
                }

                // pass 2: calculate the exp and keep track of sum
                float expsum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float expv = expf(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                // pass 3: normalize to get the softmax
                for (int t2 = 0; t2 < T; t2++) {
                    if (t2 <= t) {
                        att_bth[t2] *= expsum_inv;
                    } else {
                        // causal attention mask
                        att_bth[t2] = 0.0f;
                    }
                }

                // pass 4: accumulate weighted values into the output of attention
                float* out_bth = out + b * T * C + t * C + h * hs;
                for (int i = 0; i < hs; i++) { out_bth[i] = 0.0f; }
                for (int t2 = 0; t2 <= t; t2++) {
                    float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    float att_btht2 = att_bth[t2];
                    for (int i = 0; i < hs; i++) {
                        out_bth[i] += att_btht2 * value_t2[i];
                    }
                }
            }
        }
    }
}

int main() {
    // Define dimensions
    int B = 1;  // Batch size
    int T = 4;  // Sequence length
    int C = 8;  // Total hidden size
    int NH = 2; // Number of heads
    int hs = C / NH; // Head size
    int C3 = C * 3;  // Size of concatenated Q, K, V

    // Allocate memory
    int inp_size = B * T * C3;
    int out_size = B * T * C;
    int att_size = B * NH * T * T;

    float* inp = (float*)malloc(sizeof(float) * inp_size);
    float* out = (float*)malloc(sizeof(float) * out_size);
    float* preatt = (float*)malloc(sizeof(float) * att_size);
    float* att = (float*)malloc(sizeof(float) * att_size);

    // Initialize input tensor 'inp' with some values
    // For simplicity, let's fill it with sequential values
    for (int i = 0; i < inp_size; i++) {
        inp[i] = (float)(i + 1) / 10.0f; // Values like 0.1, 0.2, ...
    }

    // Zero initialize output and attention tensors
    for (int i = 0; i < out_size; i++) {
        out[i] = 0.0f;
    }
    for (int i = 0; i < att_size; i++) {
        preatt[i] = 0.0f;
        att[i] = 0.0f;
    }

    // Call the attention_forward function
    attention_forward(out, preatt, att, inp, B, T, C, NH);

    // Print the results
    printf("Input (inp):\n");
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            printf("Batch %d, Time %d: ", b, t);
            for (int c = 0; c < C3; c++) {
                int idx = b * T * C3 + t * C3 + c;
                printf("%.2f ", inp[idx]);
            }
            printf("\n");
        }
    }

    printf("\nOutput (out):\n");
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            printf("Batch %d, Time %d: ", b, t);
            for (int c = 0; c < C; c++) {
                int idx = b * T * C + t * C + c;
                printf("%.4f ", out[idx]);
            }
            printf("\n");
        }
    }

    printf("\nAttention Weights (att) per head:\n");
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < NH; h++) {
            printf("Batch %d, Head %d:\n", b, h);
            for (int t = 0; t < T; t++) {
                printf("Time %d -> ", t);
                for (int t2 = 0; t2 < T; t2++) {
                    int idx = b * NH * T * T + h * T * T + t * T + t2;
                    printf("%.4f ", att[idx]);
                }
                printf("\n");
            }
        }
    }

    // Free allocated memory
    free(inp);
    free(out);
    free(preatt);
    free(att);

    return 0;
}
