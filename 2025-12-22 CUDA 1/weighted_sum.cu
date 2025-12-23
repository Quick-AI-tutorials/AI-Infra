#include <iostream>
#include <cuda_runtime.h>
#include <algorithm>
#include <cuda_runtime_api.h>
#define CUDA_CHECK(expr) do {                                      \
    cudaError_t result = (expr);                                     \
    if (result != cudaSuccess) {                                     \
      fprintf(stderr, "[Printing error]: %s\n",cudaGetErrorString(result));       \
    }                                                                \
  } while (0)

__global__ void vecMult(float* w, float* x, float* res)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    res[tid] = w[tid] * x[tid];
}

int main()
{
    int vectorLength = 8192;
    int vectorSize = vectorLength * sizeof(float);

    float *x, *w, *res;
    cudaMallocHost(&x, vectorSize);
    cudaMallocHost(&w, vectorSize);
    cudaMallocHost(&res, vectorSize);

    // init the values
    std::fill(x, x + vectorLength, 4.0f); 
    std::fill(w, w + vectorLength, 0.2f);  

    float *d_x, *d_w, *d_res;
    cudaMalloc(&d_x, vectorSize);
    cudaMalloc(&d_w, vectorSize);
    cudaMalloc(&d_res, vectorSize);

    // copy to GPU
    cudaMemcpy(d_w, w, vectorSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, vectorSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, res, vectorSize, cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start);
    // 512, 256, 128 
    // nsight
    vecMult<<<8, 1024>>> (d_w, d_x, d_res); // 0.18
    // vecMult<<<512, 16>>> (d_w, d_x, d_res); // 0.22

    // Record stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %f ms\n", milliseconds);

    // Clean up events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    CUDA_CHECK(cudaGetLastError());

    // print result 
    cudaDeviceSynchronize();
    cudaMemcpy(res, d_res, vectorSize, cudaMemcpyDeviceToHost);
    for(int i = 0; i < 5; i++){
        printf("%f ", res[i]);
    }
}
