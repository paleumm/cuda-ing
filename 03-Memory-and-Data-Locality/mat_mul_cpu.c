/*
 * =====================================================================================
 *
 *       Filename:  matmul.c
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  17/07/2022 13:13:51
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Permpoon B (pb)
 *        Company:  none
 *
 * =====================================================================================
 */

#include <stdio.h>
#include <stdlib.h>

void mat_mul_host(float *h_out, float *h_mat1, float *h_mat2, int n){
    for(int i = 0 ; i < n ; i++){
        for(int j = 0 ; j < n; j++){
            float val = 0;
            for(int k = 0; k < n; k++){
                val += h_mat1[i*n + k]*h_mat2[k*n+j];
            }
            h_out[i*n+j]=val;
        }
    }
}

int main(int argc, char* argv[]){
    // executable parameter
    int n = strtol(argv[1], NULL, 10);

    int size = n*n;
    
    // initialize matrix
    float A[size], B[size], C[size];
    
    for (size_t i = 0; i < size; i++){
        B[i] = (i%n)+1;
        C[i] = (i%n)+1;
        //B[i] = 1;
        //C[i] = 1;
        //f(i%n==0) printf("\n");
        //printf("%.0f ",B[i]);
    }
    printf("\n");
    
    for(int i = 0 ; i < 100; i++){
        mat_mul_host(A, B, C, n);
    }
    //mat_mul_host(A, B, C, n);

    // printf("Output = ");
    // for( int i = 0; i < size ;i++){
    //     if(i%n==0) printf("\n");
    //     printf("%.0f ",A[i]);
    // }
    // printf("\n");

    return 0;
}