#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// ---------------------------------------------------------
// CPU sekwencyjny stencil (dla porównania)
// ---------------------------------------------------------
void cpu_stencil(const float* TAB, float* OUT, int N, int R) {
    int outN = N - 2 * R;
    for (int i = R; i < N - R; ++i) {
        for (int j = R; j < N - R; ++j) {
            float sum = 0.0f;
            for (int di = -R; di <= R; ++di)
                for (int dj = -R; dj <= R; ++dj)
                    sum += TAB[(i + di) * N + (j + dj)];
            OUT[(i - R) * outN + (j - R)] = sum;
        }
    }
}

// ---------------------------------------------------------
// GPU kernel: global memory, coalesced access
// ---------------------------------------------------------
__global__ void gpu_stencil_global(const float* TAB, float* OUT, int N, int R) {
    int i = blockIdx.y * blockDim.y + threadIdx.y; // OUT row
    int j = blockIdx.x * blockDim.x + threadIdx.x; // OUT col

    int outN = N - 2 * R;

    if (i >= outN || j >= outN) return;

    int tab_i = i + R;
    int tab_j = j + R;

    float sum = 0.0f;
    // sumowanie okna (2R+1)^2
    for (int di = -R; di <= R; ++di) {
        int row_offset = (tab_i + di) * N;
        for (int dj = -R; dj <= R; ++dj) {
            sum += TAB[row_offset + (tab_j + dj)];
        }
    }
    OUT[i * outN + j] = sum;
}

// ---------------------------------------------------------
// Inicjalizacja danych losowych
// ---------------------------------------------------------
void init_random(float* A, int size) {
    for (int i = 0; i < size; ++i)
        A[i] = (float)rand() / RAND_MAX;
}

// ---------------------------------------------------------
// Pomiar czasu GPU
// ---------------------------------------------------------
float gpu_stencil(const float* h_TAB, float* h_OUT, int N, int R) {
    int outN = N - 2 * R;
    size_t size_TAB = N * N * sizeof(float);
    size_t size_OUT = outN * outN * sizeof(float);

    float *d_TAB, *d_OUT;
    cudaMalloc(&d_TAB, size_TAB);
    cudaMalloc(&d_OUT, size_OUT);

    cudaMemcpy(d_TAB, h_TAB, size_TAB, cudaMemcpyHostToDevice);

    dim3 block(16,16);
    dim3 grid((outN + block.x -1)/block.x, (outN + block.y -1)/block.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    gpu_stencil_global<<<grid, block>>>(d_TAB, d_OUT, N, R);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_OUT, d_OUT, size_OUT, cudaMemcpyDeviceToHost);

    cudaFree(d_TAB);
    cudaFree(d_OUT);

    return ms;
}

// ---------------------------------------------------------
// Main
// ---------------------------------------------------------
int main(int argc, char** argv) {
    int N = 1024;
    int R = 1;

    if (argc >= 2) N = atoi(argv[1]);
    if (argc >= 3) R = atoi(argv[2]);
    if (N <= 2*R) { printf("Błąd: N > 2*R wymagane\n"); return 1; }

    int outN = N - 2 * R;

    float* TAB = (float*)malloc(N*N*sizeof(float));
    float* OUT_cpu = (float*)malloc(outN*outN*sizeof(float));
    float* OUT_gpu = (float*)malloc(outN*outN*sizeof(float));

    srand(time(NULL));
    init_random(TAB, N*N);

    // CPU
    clock_t t0 = clock();
    cpu_stencil(TAB, OUT_cpu, N, R);
    clock_t t1 = clock();
    double cpu_time = (double)(t1 - t0)/CLOCKS_PER_SEC;

    // GPU
    float gpu_time = gpu_stencil(TAB, OUT_gpu, N, R);

    // Sprawdzenie poprawności
    int errors = 0;
    for(int i=0;i<outN*outN;i++) {
        if(fabs(OUT_cpu[i]-OUT_gpu[i])>1e-5) errors++;
    }

    printf("OUT[0]      = %f\n", OUT_gpu[0]);
    printf("OUT[center] = %f\n", OUT_gpu[(outN/2)*outN + (outN/2)]);
    printf("CPU czas: %.3f s\n", cpu_time);
    printf("GPU czas: %.3f ms\n", gpu_time);
    printf("Błędy porównania CPU-GPU: %d\n", errors);

    free(TAB);
    free(OUT_cpu);
    free(OUT_gpu);

    return 0;
}
