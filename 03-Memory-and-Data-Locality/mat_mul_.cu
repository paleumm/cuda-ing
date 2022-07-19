/*
 * =====================================================================================
 *
 *       Filename:  mat_mul_.cu
 *    Description:  matrix multiplication with any dimension
 *
 *        Version:  1.0
 *        Created:  17/07/2022 19:50:57
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

#define TILE_WIDTH 16

__global__ void mat_mul_tiles(float *d_M, float *d_N, float *d_P, int width){

    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float Pval = 0;

    // Strip-mining => break into phases
    for(int tile = 0; tile < ceil(width/(float)TILE_WIDTH); tile++){
        // Assign the tiles for each thread. 
        // d_M(y,x) => Mds(y,x), d_N(y,x) => Nds(y,x)
        if((row < width) && (tile * TILE_WIDTH + tx)<width)
            Mds[ty][tx] = d_M[row * width + tile * TILE_WIDTH + tx];
        if((tile*TILE_WIDTH + ty) < width && (col < width))
            Nds[ty][tx] = d_N[(tile * TILE_WIDTH +ty)*width + col];

        __syncthreads(); // wait for all thread to load tile

        // Mds(y,k)*Nds(k,x) => Pval
        for(int k = 0; k < TILE_WIDTH; k++){
            Pval += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads(); // ensure that all thread finished using tile
    }

    if((row < width) && (col < width)) d_P[row*width + col] = Pval;
}

__host__ void mul_mat(float *h_out, float *h_mat1, float *h_mat2, int n){
    // size of matrix
    int mat_sz = n * n * sizeof(float);
    float *d_first, *d_second, *d_out;

    cudaError_t errFirst = cudaMalloc((void**)&d_first, mat_sz);
    if(errFirst != cudaSuccess){
       printf("%s in %s at line %d\n",cudaGetErrorString(errFirst), __FILE__, __LINE__);
       exit(EXIT_FAILURE);
    }
    cudaMemcpy(d_first, h_mat1, mat_sz, cudaMemcpyHostToDevice);

    cudaError_t errSecond = cudaMalloc((void**)&d_second, mat_sz);
    if(errSecond != cudaSuccess){
       printf("%s in %s at line %d\n",cudaGetErrorString(errSecond), __FILE__, __LINE__);
       exit(EXIT_FAILURE);
    }
    cudaMemcpy(d_second, h_mat2, mat_sz, cudaMemcpyHostToDevice);

    cudaError_t errOut = cudaMalloc((void**)&d_out, mat_sz);  
    if(errOut != cudaSuccess){
       printf("%s in %s at line %d\n",cudaGetErrorString(errOut), __FILE__, __LINE__);
       exit(EXIT_FAILURE);
    }

    dim3 dimBlock(TILE_WIDTH , TILE_WIDTH);
    dim3 dimGrid(ceil(n/float(dimBlock.x)), ceil(n/float(dimBlock.y)));
    mat_mul_tiles<<<dimGrid, dimBlock>>>(d_first, d_second, d_out, n);
    cudaMemcpy(h_out, d_out, mat_sz, cudaMemcpyDeviceToHost);

    cudaFree(d_first); cudaFree(d_second); cudaFree(d_out);

}

int main(int argc, char* argv[]){
    // code section
    

    return 0;
}