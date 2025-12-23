# CUDA Video 1

CUDA code to calculate vector mutiplication

```cpp
__global__ void vecMult(float* w, float* x, float* res)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    res[tid] = w[tid] * x[tid];
}

vecMult<<<8, 1024>>> (d_w, d_x, d_res); 
```

Full code in `weighted_sum.cu`.
