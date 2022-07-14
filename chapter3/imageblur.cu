/*
 * =====================================================================================
 *
 *       Filename:  imageblur.cu
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  15/07/2022 00:41:21
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

// 1 => 3 x 3 kernel
#define BLUR_SIZE 1

__global__ void blurKernel(unsigned char *in, unsigned char *out, int w, int h){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(col < w && row < h){
        int pixval = 0;
        int pixels = 0;

        for(int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow){
            for(int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol){
                int curRow = row + blurRow;
                int curCol = col + blurCol;
                
                // edge case when curRow or curCol may gather the outer location of the image.
                if(curRow > -1 && curRow < h && curCol > -1 && curCol < w){
                    pixval += in[curRow * w + curCol];
                    pixels++;
                }
            }
        }
        out[row*w + col] = (unsigned char)(pixval / pixels);
    }
}

int main(int argc, char* argv[]){
    // code section
    

    return 0;
}