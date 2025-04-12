// matrix_mul_cpu.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <windows.h>

// Function to get time in milliseconds using Windows API
double get_time_ms() {
    LARGE_INTEGER frequency;
    LARGE_INTEGER counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1000.0 / (double)frequency.QuadPart;
}

int main() {
    const int N = 32;
    const int SIZE = N * N;
    
    // Start overall timing
    double overall_start = get_time_ms();
    
    // Allocate memory for matrices
    float *a_data = (float*)malloc(SIZE * sizeof(float));
    float *b_data = (float*)malloc(SIZE * sizeof(float));
    float *c_data = (float*)malloc(SIZE * sizeof(float));
    
    // Initialize matrices with the same values as the Rust version
    for (int i = 0; i < SIZE; i++) {
        a_data[i] = (float)i;
        b_data[i] = (float)i;
    }
    
    // Matrix multiplication on CPU
    double execution_start = get_time_ms();
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += a_data[row * N + k] * b_data[k * N + col];
            }
            c_data[row * N + col] = sum;
        }
    }
    double execution_time = get_time_ms() - execution_start;
    printf("Kernel execution time: %.3f ms\n", execution_time);
    
    // Verify output
    int errors = 0;
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            float expected = 0.0f;
            for (int k = 0; k < N; k++) {
                expected += a_data[row * N + k] * b_data[k * N + col];
            }
            
            int idx = row * N + col;
            if (fabs(c_data[idx] - expected) > 1e-3) {
                errors++;
            }
        }
    }
    
    if (errors == 0) {
        printf("Matmul done. All results correct.\n");
    } else {
        printf("Matmul done with %d errors.\n", errors);
    }
    
    // Log total time
    double total_time = get_time_ms() - overall_start;
    printf("Total execution time: %.3f ms\n", total_time);
    
    // Free memory
    free(a_data);
    free(b_data);
    free(c_data);
    
    return 0;
}