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

__global__ void add_mat_kernel_0(float *a, float *b, float *c, int n){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int output_index = row * n + col;
    if(output_index < n*n){
        c[output_index] = a[output_index] + b[output_index];
    }
}

__global__ void add_mat_kernel_1(float *a, float *b, float *c, int n){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < n*n) c[index] = a[index] + b[index]; 
}

// __global__ void add_mat_kernel_2(float *a, float *b, float *c, int n){
//     int index = blockIdx.x * blockDim.x + threadIdx.x;
//     if(index < n)
//         for (size_t i = 0; i < n; ++i)
//             c[index+i] = a[index+i] + b[index+i];
// }


/*
 * =====================================================================================
 *
 *       Function:  add_matrix
 *    Description:  
 *     Parameters:  n[dimension of square matrix], mode[0:normal, 1:thread->element,
 *                  2:thread->row, 3:thread:col]
 *
 * =====================================================================================
 */
__host__ void add_matrix(float *h_out, float *h_first, float *h_second, int n, int mode){
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

    switch (mode)
    {
    case 1:
        add_mat_kernel_1<<<ceil(n/64.0),64>>>(d_first, d_second, d_out, n);
        break;
    case 2:
        // add_mat_kernel_2<<<ceil(n/64.0),64>>>(d_first, d_second, d_out, n);
        break;
    case 3:
        break;
    default:
        add_mat_kernel_0<<<ceil(n/64.0),64>>>(d_first, d_second, d_out, n);
        break;
    }

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    cudaFree(d_first); cudaFree(d_second); cudaFree(d_out);

}

int main(int argc, char* argv[]){
    // code section
    int n = strtol(argv[1], NULL, 10);
    unsigned short mode = strtol(argv[2], NULL, 10);

    if(mode > 3){printf("mode should be 0-3\n");return 1;}

    int size = n*n;

    // initialize matrix
    float A[size], B[size], C[size];
    for (size_t i = 0; i < size; i++)
    {
        A[i] = i % 10;
        B[i] = (i+1) % 10;
    }
    //for (int i = 0; i < size; i++) printf("A[%d] = %f, B[%d]=%f\n",i,A[i],i,B[i]);
    
    add_matrix(C, A, B, n, mode);
    for (size_t i = 0; i < size; i++)
    {
        printf("%.0f + %.0f = %.0f\n", A[i], B[i], C[i]);
    }
    return 0;
}