#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// The layernorm_forward function as provided
void layernorm_forward(float* out, float* mean, float* rstd,
                       float* inp, float* weight, float* bias,
                       int B, int T, int C) {
    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // Seek to the input position inp[b,t,:]
            float* x = inp + b * T * C + t * C;
            // Calculate the mean
            float m = 0.0f;
            for (int i = 0; i < C; i++) {
                m += x[i];
            }
            m = m / C;
            // Calculate the variance (without any bias correction)
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v / C;
            // Calculate the rstd
            float s = 1.0f / sqrtf(v + eps);
            // Seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
                float n = s * (x[i] - m); // Normalized output
                float o = n * weight[i] + bias[i]; // Scale and shift it
                out_bt[i] = o; // Write
            }
            // Cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

int main() {
    // Define dimensions
    int B = 2; // Batch size
    int T = 3; // Sequence length
    int C = 4; // Number of features

    // Calculate total sizes
    int inp_size = B * T * C;
    int param_size = C;
    int stat_size = B * T;

    // Allocate memory
    float* inp = (float*)malloc(sizeof(float) * inp_size);
    float* weight = (float*)malloc(sizeof(float) * param_size);
    float* bias = (float*)malloc(sizeof(float) * param_size);
    float* out = (float*)malloc(sizeof(float) * inp_size);
    float* mean = (float*)malloc(sizeof(float) * stat_size);
    float* rstd = (float*)malloc(sizeof(float) * stat_size);

    // Initialize input tensor 'inp' with some values
    for (int i = 0; i < inp_size; i++) {
        inp[i] = (float)(i + 1); // Fill with values 1, 2, 3, ...
    }

    // Initialize 'weight' to ones
    for (int i = 0; i < param_size; i++) {
        weight[i] = 1.0f;
    }

    // Initialize 'bias' to zeros
    for (int i = 0; i < param_size; i++) {
        bias[i] = 0.0f;
    }

    // Call the layernorm_forward function
    layernorm_forward(out, mean, rstd, inp, weight, bias, B, T, C);

    // Print the results
    printf("Input (inp):\n");
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            printf("Batch %d, Time %d: ", b, t);
            for (int c = 0; c < C; c++) {
                int idx = b * T * C + t * C + c;
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

    printf("\nMean (mean):\n");
    for (int b = 0; b < B; b++) {
        printf("Batch %d: ", b);
        for (int t = 0; t < T; t++) {
            int idx = b * T + t;
            printf("%.4f ", mean[idx]);
        }
        printf("\n");
    }

    printf("\nReciprocal Std Dev (rstd):\n");
    for (int b = 0; b < B; b++) {
        printf("Batch %d: ", b);
        for (int t = 0; t < T; t++) {
            int idx = b * T + t;
            printf("%.4f ", rstd[idx]);
        }
        printf("\n");
    }

    // Free allocated memory
    free(inp);
    free(weight);
    free(bias);
    free(out);
    free(mean);
    free(rstd);

    return 0;
}
