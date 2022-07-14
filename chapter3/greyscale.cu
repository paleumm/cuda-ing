/*
 * =====================================================================================
 *
 *       Filename:  greyscale.cu
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  15/07/2022 00:40:33
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

#define CHANNELS 3

__global__ void colorToGreyscaleConversion(unsigned char *Pout, unsigned char *Pin, int width, int height){
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    if(col < width && row < height){
        int greyOffset = row * width + col;
        int rgbOffset = greyOffset * CHANNELS;

        unsigned char r = Pin[rgbOffset];
        unsigned char g = Pin[rgbOffset+1];
        unsigned char b = Pin[rgbOffset+2];

        Pout[greyOffset] = 0.21f*r + 0.72f*g + 0.07f*b;
    }
}

int main(int argc, char* argv[]){
    // code section
    

    return 0;
}