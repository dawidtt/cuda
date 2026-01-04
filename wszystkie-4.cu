#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define BS 16

// ----------------------------------------------------------------------
// CPU Stencil - Referencja
// ----------------------------------------------------------------------
void cpu_stencil(const float* TAB, float* OUT, int N, int R) {
    int outN = N - 2 * R;
    for (int i = 0; i < outN; ++i) {
        for (int j = 0; j < outN; ++j) {
            float sum = 0.0f;
            for (int di = -R; di <= R; ++di) {
                for (int dj = -R; dj <= R; ++dj) {
                    sum += TAB[(i + R + di) * N + (j + R + dj)];
                }
            }
            OUT[i * outN + j] = sum;
        }
    }
}

// ----------------------------------------------------------------------
// (a) Global Coalesced
// ----------------------------------------------------------------------
__global__ void kern_a_coalesced(const float* TAB, float* OUT, int N, int R, int k) {
    int outN = N - 2 * R;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j_start = (blockIdx.x * blockDim.x + threadIdx.x) * k;

    if (i < outN) {
        for (int kk = 0; kk < k; kk++) {
            int j = j_start + kk;
            if (j < outN) {
                float sum = 0.0f;
                for (int di = -R; di <= R; di++) {
                    for (int dj = -R; dj <= R; dj++) {
                        sum += TAB[(i + R + di) * N + (j + R + dj)];
                    }
                }
                OUT[i * outN + j] = sum;
            }
        }
    }
}

// ----------------------------------------------------------------------
// (b) Global Uncoalesced
// ----------------------------------------------------------------------
__global__ void kern_b_uncoalesced(const float* TAB, float* OUT, int N, int R, int k) {
    int outN = N - 2 * R;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i_start = (blockIdx.y * blockDim.y + threadIdx.y) * k;

    if (j < outN) {
        for (int kk = 0; kk < k; kk++) {
            int i = i_start + kk;
            if (i < outN) {
                float sum = 0.0f;
                for (int di = -R; di <= R; di++) {
                    for (int dj = -R; dj <= R; dj++) {
                        sum += TAB[(i + R + di) * N + (j + R + dj)];
                    }
                }
                OUT[i * outN + j] = sum;
            }
        }
    }
}

// ----------------------------------------------------------------------
// (c) Shared Memory Efficient (z dynamicznym pitch)
// ----------------------------------------------------------------------
__global__ void kern_c_shared(const float* TAB, float* OUT, int N, int R, int k) {
    extern __shared__ float sdata[];
    
    const int outN = N - 2 * R;
    const int tile_w = blockDim.x * k + 2 * R;
    const int tile_h = blockDim.y + 2 * R;
    const int pitch = (tile_w + 31) & ~31; // Wyrównanie do 32 elementów (128 bajtów)

    const int global_tile_start_i = blockIdx.y * blockDim.y;
    const int global_tile_start_j = blockIdx.x * blockDim.x * k;

    for (int si = threadIdx.y; si < tile_h; si += blockDim.y) {
        for (int sj = threadIdx.x; sj < tile_w; sj += blockDim.x) {
            int gi = global_tile_start_i + si; 
            int gj = global_tile_start_j + sj;
            if (gi < N && gj < N)
                sdata[si * pitch + sj] = TAB[gi * N + gj];
            else
                sdata[si * pitch + sj] = 0.0f;
        }
    }
    __syncthreads();

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int res_i = global_tile_start_i + ty;

    if (res_i < outN) {
        for (int kk = 0; kk < k; kk++) {
            const int res_j = global_tile_start_j + tx * k + kk;
            if (res_j < outN) {
                float sum = 0.0f;
                for (int di = 0; di <= 2 * R; di++) {
                    int row_addr = (ty + di) * pitch;
                    for (int dj = 0; dj <= 2 * R; dj++) {
                        sum += sdata[row_addr + (tx * k + kk + dj)];
                    }
                }
                OUT[res_i * outN + res_j] = sum;
            }
        }
    }
}

// ----------------------------------------------------------------------
// (d) Shared Memory Bank Conflicts
// ----------------------------------------------------------------------
__global__ void kern_d_conflicts(const float* TAB, float* OUT, int N, int R, int k) {
    extern __shared__ float sdata[];
    const int outN = N - 2 * R;
    const int tile_w = blockDim.x * k + 2 * R;
    const int tile_h = blockDim.y + 2 * R;
    const int pitch = (tile_w + 31) & ~31;

    const int global_tile_start_i = blockIdx.y * blockDim.y;
    const int global_tile_start_j = blockIdx.x * blockDim.x * k;

    for (int si = threadIdx.y; si < tile_h; si += blockDim.y) {
        for (int sj = threadIdx.x; sj < tile_w; sj += blockDim.x) {
            int gi = global_tile_start_i + si; 
            int gj = global_tile_start_j + sj;
            if (gi < N && gj < N)
                sdata[si * pitch + sj] = TAB[gi * N + gj];
            else
                sdata[si * pitch + sj] = 0.0f;
        }
    }
    __syncthreads();

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;

    if (global_tile_start_i + ty < outN) {
        for (int kk = 0; kk < k; kk++) {
            const int res_j = global_tile_start_j + tx * k + kk;
            if (res_j < outN) {
                float sum = 0.0f;
                for (int di = 0; di <= 2 * R; di++) {
                    int row_addr = (ty + di) * pitch;
                    for (int dj = 0; dj <= 2 * R; dj++) {
                        // Sztuczny konflikt: wiele wątków z warp czyta ten sam bank
                        volatile float* vs = sdata;
                        sum += vs[(tx * 32) % (pitch * tile_h)] * 1e-10f; 
                        sum += sdata[row_addr + (tx * k + kk + dj)];
                    }
                }
                OUT[(global_tile_start_i + ty) * outN + res_j] = sum;
            }
        }
    }
}

void validate(const char* label, float* cpu, float* gpu, int N, int R, float ms) {
    int outN = N - 2 * R;
    int err = 0;
    for (int i = 0; i < outN * outN; i++) {
        if (fabs(cpu[i] - gpu[i]) > 0.1f) err++;
    }
    double ops = (double)outN * outN * (2 * R + 1) * (2 * R + 1);
    double gflops = (ops / (ms / 1000.0)) / 1e9;
    printf("%-15s | %8.3f ms | %7.2f GFLOPS | Błędy: %d\n", label, ms, gflops, err);
}

int main(int argc, char** argv) {
    int N = 4096, R = 4, k = 1;
    if (argc >= 2) N = atoi(argv[1]);
    if (argc >= 3) R = atoi(argv[2]);
    if (argc >= 4) k = atoi(argv[3]);

    int outN = N - 2 * R;
    size_t sizeIn = (size_t)N * N * sizeof(float);
    size_t sizeOut = (size_t)outN * outN * sizeof(float);

    float *h_TAB = (float*)malloc(sizeIn);
    float *h_OUT_cpu = (float*)malloc(sizeOut);
    float *h_OUT_gpu = (float*)malloc(sizeOut);
    
    srand(42);
    for (size_t i = 0; i < (size_t)N * N; i++) h_TAB[i] = (float)rand() / RAND_MAX;

    float *d_TAB, *d_OUT;
    cudaMalloc(&d_TAB, sizeIn);
    cudaMalloc(&d_OUT, sizeOut);
    cudaMemcpy(d_TAB, h_TAB, sizeIn, cudaMemcpyHostToDevice);

    printf("Pomiary: N=%d, R=%d, k=%d, BS=%dx%d\n", N, R, k, BS, BS);
    printf("----------------------------------------------------------------------\n");

    clock_t s_cpu = clock();
    cpu_stencil(h_TAB, h_OUT_cpu, N, R);
    double cpu_ms = (double)(clock() - s_cpu) * 1000.0 / CLOCKS_PER_SEC;
    printf("CPU             | %8.3f ms | %7.2f GFLOPS\n", cpu_ms, ((double)outN*outN*(2*R+1)*(2*R+1)/(cpu_ms/1000.0))/1e9);

    dim3 block(BS, BS);
    dim3 grid((outN / k + BS - 1) / BS, (outN + BS - 1) / BS);
    
    // Obliczanie rozmiaru pamięci współdzielonej z wyrównaniem (pitch)
    int tw = BS * k + 2 * R;
    int th = BS + 2 * R;
    int pitch = (tw + 31) & ~31;
    size_t sh_size = (size_t)pitch * th * sizeof(float);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    float ms;

    // (a)
    cudaEventRecord(start);
    kern_a_coalesced<<<grid, block>>>(d_TAB, d_OUT, N, R, k);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    cudaMemcpy(h_OUT_gpu, d_OUT, sizeOut, cudaMemcpyDeviceToHost);
    validate("GPU Coalesced", h_OUT_cpu, h_OUT_gpu, N, R, ms);

    // (b)
    cudaEventRecord(start);
    kern_b_uncoalesced<<<grid, block>>>(d_TAB, d_OUT, N, R, k);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    cudaMemcpy(h_OUT_gpu, d_OUT, sizeOut, cudaMemcpyDeviceToHost);
    validate("GPU Uncoalesced", h_OUT_cpu, h_OUT_gpu, N, R, ms);

    // (c)
    cudaEventRecord(start);
    kern_c_shared<<<grid, block, sh_size>>>(d_TAB, d_OUT, N, R, k);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    cudaMemcpy(h_OUT_gpu, d_OUT, sizeOut, cudaMemcpyDeviceToHost);
    validate("GPU Shared", h_OUT_cpu, h_OUT_gpu, N, R, ms);

    // (d)
    cudaEventRecord(start);
    kern_d_conflicts<<<grid, block, sh_size>>>(d_TAB, d_OUT, N, R, k);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    cudaMemcpy(h_OUT_gpu, d_OUT, sizeOut, cudaMemcpyDeviceToHost);
    validate("GPU Conflicts", h_OUT_cpu, h_OUT_gpu, N, R, ms);

    cudaFree(d_TAB); cudaFree(d_OUT);
    free(h_TAB); free(h_OUT_cpu); free(h_OUT_gpu);
    return 0;
}