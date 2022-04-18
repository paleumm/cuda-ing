
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void add_vector(int *a, int *b, int *c)
{
    int index = threadIdx.x;
    c[index] = a[index] + b[index];
    return;
}

int main()
{
    int a[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int b[] = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13};

    int c[sizeof(a) / sizeof(int)] = {0};

    int *cuda_a = 0;
    int *cuda_b = 0;
    int *cuda_c = 0;

    // allocate memory in GPU
    cudaMalloc(&cuda_a, sizeof(a));
    cudaMalloc(&cuda_b, sizeof(b));
    cudaMalloc(&cuda_c, sizeof(c));

    // copy vector to GPU's memory
    cudaMemcpy(cuda_a, a, sizeof(a), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_b, b, sizeof(b), cudaMemcpyHostToDevice);

    // add_vector <<< GRID_SIZE, BLOCK_SIZE >>> (parameters);
    add_vector<<<1, sizeof(a) / sizeof(int)>>>(cuda_a, cuda_b, cuda_c);

    cudaMemcpy(c, cuda_c, sizeof(c), cudaMemcpyDeviceToHost);

    return;
}