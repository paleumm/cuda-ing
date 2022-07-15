/*
 * =====================================================================================
 *
 *       Filename:  add_vector.cu
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  15/07/2022 00:38:22
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  Permpoon B (pb)
 *        Company:  none
 *
 * =====================================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

__global__ void add_vec_kernel(float *A, float *B, float *C, int n){
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < n) C[index] = A[index] + B[index];
}
__host__ void add_vec(float *h_A, float *h_B, float *h_C, int n){
    int size = n*sizeof(float);
    float *d_A, *d_B, *d_C;

    cudaError_t errA = cudaMalloc((void**)&d_A, size);
    if(errA != cudaSuccess){
        printf("%s in %s at line %d\n",cudaGetErrorString(errA), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    cudaError_t errB = cudaMalloc((void**)&d_B, size);
    if(errB != cudaSuccess){
       printf("%s in %s at line %d\n",cudaGetErrorString(errB), __FILE__, __LINE__);
       exit(EXIT_FAILURE);
    }
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    cudaError_t errC = cudaMalloc((void**)&d_C, size);
    if(errC != cudaSuccess){
       printf("%s in %s at line %d\n",cudaGetErrorString(errC), __FILE__, __LINE__);
       exit(EXIT_FAILURE);
    }

    // kernel invocation
    add_vec_kernel<<<ceil(n/1024.0), 1024>>>(d_A, d_B, d_C, n);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);cudaFree(d_B);cudaFree(d_C);
}

int main(int argc, char* argv[]){
    // size of vector
    int n = strtol(argv[1], NULL, 10);

    // init vector
    float A[n], B[n], C[n];
    for (size_t i = 0; i < n; i++)
    {
        A[i] = i % 10;
        B[i] = (i+1) % 10;
    }

    add_vec(A, B, C, n);
    for (size_t i = 0; i < 20; i++)
    {
        printf("%lf\n", C[i]);
    }
    
    return 0;
}