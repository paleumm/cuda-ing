/*
 * =====================================================================================
 *
 *       Filename:  mul_matrix_vector.cu
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  15/07/2022 02:22:15
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

__global__ void mul_mat_vec_kernel(float *a, float *b, float *c, int n){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    //if(col < n && row < n){
        float sum = 0;
        for (size_t i = 0; i < n; i++)
            sum += b[row * n + i]*c[i];
        a[col] = sum;
    //}
}

__host__ void mul_mat_vec(float *h_out, float *h_mat, float *h_vec, int n){
    // size of matrix
    int mat_sz = n * n * sizeof(float);
    int vec_sz = n * sizeof(float);

    int mat_dim = n * n;
    float *d_first, *d_second, *d_out;

    cudaError_t errFirst = cudaMalloc((void**)&d_first, mat_sz);
    if(errFirst != cudaSuccess){
       printf("%s in %s at line %d\n",cudaGetErrorString(errFirst), __FILE__, __LINE__);
       exit(EXIT_FAILURE);
    }
    cudaMemcpy(d_first, h_mat, mat_sz, cudaMemcpyHostToDevice);

    cudaError_t errSecond = cudaMalloc((void**)&d_second, vec_sz);
    if(errSecond != cudaSuccess){
       printf("%s in %s at line %d\n",cudaGetErrorString(errSecond), __FILE__, __LINE__);
       exit(EXIT_FAILURE);
    }
    cudaMemcpy(d_second, h_vec, vec_sz, cudaMemcpyHostToDevice);

    cudaError_t errOut = cudaMalloc((void**)&d_out, vec_sz);  
    if(errOut != cudaSuccess){
       printf("%s in %s at line %d\n",cudaGetErrorString(errOut), __FILE__, __LINE__);
       exit(EXIT_FAILURE);
    }

    mul_mat_vec_kernel<<<ceil(mat_dim/1024.0), 1024>>>(d_out, d_first, d_second, n);

    cudaMemcpy(h_out, d_out, vec_sz, cudaMemcpyDeviceToHost);

    cudaFree(d_first); cudaFree(d_second); cudaFree(d_out);

}

int main(int argc, char* argv[]){
    // code section
    int n = strtol(argv[1], NULL, 10);

    int size = n*n;
    
    // initialize matrix
    float A[n], B[size], C[n];
    //float B[] = {4,5,6,7,1,2,3,4,4,5,6,7,1,2,3,4};
    for (size_t i = 0; i < size; i++){
        B[i] = i % 8;
        if(i%n==0) printf("\n");
        printf("%.0f ",B[i]);
    }
    printf("\n");
    printf("\n");
    
    for (size_t i = 0; i < n; i++){
        C[i] = i % 8;
        printf("%.0f ",C[i]);
    }
    printf("\n");
    

    mul_mat_vec(A, B, C, n);

    for( int i = 0; i < n ;i++){
        printf("%.0f ",A[i]);
    }
    printf("\n");

    return 0;
}