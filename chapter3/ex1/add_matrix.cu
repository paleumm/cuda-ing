/*
 * =====================================================================================
 *
 *       Filename:  add_matrix.cu
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  15/07/2022 00:42:28
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

__global__ void add_mat_kernel(float *a, float *b, float *c, int n){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int output_index = row * n + col;
    if(output_index < n*n){
        c[output_index] = a[output_index] + b[output_index];
    }
}

__host__ void add_matrix(float *h_out, float *h_first, float *h_second, int n){
    // size of matrix
    int size = n * n * sizeof(float);
    float *d_first, *d_second, *d_out;

    cudaError_t errFirst = cudaMalloc((void**)&d_first, size);
    if(errFirst != cudaSuccess){
       printf("%s in %s at line %d\n",cudaGetErrorString(errFirst), __FILE__, __LINE__);
       exit(EXIT_FAILURE);
    }
    cudaMemcpy(d_first, h_first, size, cudaMemcpyHostToDevice);

    cudaError_t errSecond = cudaMalloc((void**)&d_second, size);
    if(errSecond != cudaSuccess){
       printf("%s in %s at line %d\n",cudaGetErrorString(errSecond), __FILE__, __LINE__);
       exit(EXIT_FAILURE);
    }
    cudaMemcpy(d_second, h_second, size, cudaMemcpyHostToDevice);

    cudaError_t errOut = cudaMalloc((void**)&d_out, size);  
    if(errOut != cudaSuccess){
       printf("%s in %s at line %d\n",cudaGetErrorString(errOut), __FILE__, __LINE__);
       exit(EXIT_FAILURE);
    }

    add_mat_kernel<<<ceil(n/256.0),256>>>(d_first, d_second, d_out, n);

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    cudaFree(d_first); cudaFree(d_second); cudaFree(d_out);

}

int main(int argc, char* argv[]){
    // code section
    int n = strtol(argv[1], NULL, 10);

    int size = n*n;
    float A[size], B[size], C[size];
    for (size_t i = 0; i < size; i++)
    {
        A[i] = i % 10;
        B[i] = (i+1) % 10;
    }
    //for (int i = 0; i < size; i++) printf("A[%d] = %f, B[%d]=%f\n",i,A[i],i,B[i]);
    
    add_matrix(C, A, B, n);
    for (size_t i = 0; i < size; i++)
    {
        printf("%f + %f = %f\n", A[i], B[i], C[i]);
    }
    return 0;
}