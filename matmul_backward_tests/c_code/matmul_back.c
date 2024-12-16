#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <stdbool.h>


void matmul_forward(float* out,
                    float* inp, float* weight, float* bias,
                    int B, int T, int C, int OC) {
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)
    // This is Karpathy code
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* out_bt = out + b * T * OC + t * OC;
            float* inp_bt = inp + b * T * C + t * C;
            for (int o = 0; o < OC; o++) {
                float val = (bias != NULL) ? bias[o] : 0.0f;
                float* wrow = weight + o * C;
                for (int i = 0; i < C; i++) {
                    val += inp_bt[i] * wrow[i];
                }
                out_bt[o] = val;
            }
        }
    }
}

void matmul_backward(float* dinp, float* dweight, float* dbias,
                     float* dout, float* inp, float* weight,
                     int B, int T, int C, int OC) {
    // most of the running time is spent here and in matmul_forward
    // this backward could be done in a single "round" of loops
    // but that doesn't afford an efficient parallelization strategy

    // backward into inp first, parallelize over B,T
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dout_bt = dout + b * T * OC + t * OC;
            float* dinp_bt = dinp + b * T * C + t * C;
            for (int o = 0; o < OC; o++) {
                float* wrow = weight + o*C;
                float d = dout_bt[o];
                for (int i = 0; i < C; i++) {
                    dinp_bt[i] += wrow[i] * d;
                }
            }
        }
    }
    // backward into weight/bias, parallelize over output channels OC
    #pragma omp parallel for
    for (int o = 0; o < OC; o++) {
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                float* dout_bt = dout + b * T * OC + t * OC;
                float* inp_bt = inp + b * T * C + t * C;
                float* dwrow = dweight + o*C;
                float d = dout_bt[o];
                if (dbias != NULL) { dbias[o] += d; }
                for (int i = 0; i < C; i++) {
                    dwrow[i] += inp_bt[i] * d;
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    bool debug = false;
    if (argc != 5) {
        fprintf(stderr, "Usage: %s B T C OC\n", argv[0]);
        return 1;
    }
    // Define dimensions
    char* endptr;
    int B = (int)strtol(argv[1], &endptr, 10);
    if (*endptr != '\0' || B <= 0) {
        fprintf(stderr, "Invalid value for B: %s\n", argv[1]);
        return 1;
    }

    int T = (int)strtol(argv[2], &endptr, 10);
    if (*endptr != '\0' || T <= 0) {
        fprintf(stderr, "Invalid value for T: %s\n", argv[2]);
        return 1;
    }

    int C = (int)strtol(argv[3], &endptr, 10);
    if (*endptr != '\0' || C <= 0) {
        fprintf(stderr, "Invalid value for C: %s\n", argv[3]);
        return 1;
    }

    int OC = (int)strtol(argv[4], &endptr, 10);
    if (*endptr != '\0' || OC <= 0) {
        fprintf(stderr, "Invalid value for OC: %s\n", argv[4]);
        return 1;
    }
    // Allocate memory for input, weight, bias, and output
    float* inp = (float*)malloc(B * T * C * sizeof(float));
    float* weight = (float*)malloc(OC * C * sizeof(float));
    float* bias = (float*)malloc(OC * sizeof(float));
    float* out = (float*)malloc(B * T * OC * sizeof(float));

    // Check for successful memory allocation
    if (inp == NULL || weight == NULL || bias == NULL || out == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        return 1;
    }
    printf("Parameters: B=%d, T=%d, C=%d, OC=%d\n", B, T, C, OC);

    // Initialize input tensor (B x T x C)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int c_idx = 0; c_idx < C; c_idx++) {
                int idx = b * T * C + t * C + c_idx;
                inp[idx] = (float)(idx % 10) * 0.1f;  // Sample data
            }
        }
    }

    // Initialize weight matrix (OC x C)
    for (int o = 0; o < OC; o++) {
        for (int c_idx = 0; c_idx < C; c_idx++) {
            int idx = o * C + c_idx;
            weight[idx] = (float)((idx % 5) * 0.1f + 0.5f);  // Sample data
        }
    }

    // Initialize bias vector (OC)
    for (int o = 0; o < OC; o++) {
        bias[o] = 0.1f;  // Sample bias
    }
    if (debug) {
        printf("Input tensors: \n");
        for (int b = 0; b < B; b++) {
            printf("Batch %d:\n", b);
            for (int t = 0; t < T; t++) {
                printf("  Time step %d: ", t);
                for (int c = 0; c < C; c++) {
                    int idx = b * T * C + t * C + c;
                    printf("%.2f ", inp[idx]);
                }
                printf("\n");
            }
        }
        printf("Input weights\n");
        for (int o = 0; o < OC; o++) {
            printf("Output channel %d: ", o);
            for (int c = 0; c < C; c++) {
                int idx = o * C + c;
                printf("%.2f ", weight[idx]);
            }
            printf("\n");
        }

        printf("Input bias\n");
        for (int o = 0; o < OC; o++) {
            printf("%.2f ", bias[o]);
        }
    }

    // Call matmul_forward
    double start_time = omp_get_wtime();
    matmul_forward(out, inp, weight, bias, B, T, C, OC);
    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;
    printf("Time taken by matmul_forward: %.6f seconds\n", elapsed_time);
    // Print the output tensor

    if (debug) {
        printf("\nOutput Tensor:\n");
        for (int b = 0; b < B; b++) {
            printf("Batch %d:\n", b);
            for (int t = 0; t < T; t++) {
                printf("  Time step %d: ", t);
                for (int o = 0; o < OC; o++) {
                    int idx = b * T * OC + t * OC + o;
                    printf("%.2f ", out[idx]);
                }
                printf("\n");
            }
        }
    }
    // compute the backward
    float* dinp = (float*)malloc(B * T * C * sizeof(float));
    float* dweight = (float*)malloc(OC * C * sizeof(float));
    float* dbias = (float*)malloc(OC * sizeof(float));
    float* dout = (float*)malloc(B * T * OC * sizeof(float));
    if (dinp == NULL || dweight == NULL || dbias == NULL || dout == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        return 1;
    }
    // Initialize dout tensor (B x T x OC)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int o = 0; o < OC; o++) {
                int idx = b * T * OC + t * OC + o;
                dout[idx] = (float)(idx % 10) * 0.1f;  // Sample data
            }
        }
    }
    // Call matmul_backward
    start_time = omp_get_wtime();
    matmul_backward(dinp, dweight, dbias, dout, inp, weight, B, T, C, OC);
    end_time = omp_get_wtime();
    elapsed_time = end_time - start_time;
    printf("Time taken by matmul_backward: %.6f seconds\n", elapsed_time);
    if (debug) {
        printf("\ndOutput Tensor:\n");
        for (int b = 0; b < B; b++) {
            printf("Batch %d:\n", b);
            for (int t = 0; t < T; t++) {
                printf("  Time step %d: ", t);
                for (int o = 0; o < OC; o++) {
                    int idx = b * T * OC + t * OC + o;
                    printf("%.2f ", dout[idx]);
                }
                printf("\n");
            }
        }
    }

    // Free allocated memory
    free(inp);
    free(weight);
    free(bias);
    free(out);

    return 0;
}
