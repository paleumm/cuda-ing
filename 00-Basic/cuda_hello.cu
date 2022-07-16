/*
 * =====================================================================================
 *
 *       Filename:  cuda_hello.cu
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  15/07/2022 00:37:01
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

__global__ void Hello(void){
    printf("Hello from thread %d in block %d\n", threadIdx.x, blockIdx.x);
}

int main(int argc, char* argv[]){
    // code section
    int block_ct;
    int thd_per_block;

    block_ct = strtol(argv[1], NULL, 10);
    thd_per_block = strtol(argv[2], NULL, 10);

    Hello<<<block_ct, thd_per_block>>>();

    cudaDeviceSynchronize();

    return 0;
}