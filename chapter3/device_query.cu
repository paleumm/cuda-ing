/*
 * =====================================================================================
 *
 *       Filename:  device_query.cu
 *    Description:  querying device properties
 *
 *        Version:  1.0
 *        Created:  15/07/2022 00:39:41
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  Permpoon B (pb)
 *        Company:  none
 *
 * =====================================================================================
 */

#include <iostream>
#include <stdlib.h>
#include <cuda.h>

__global__ void func(void);

int main(int argc, char* argv[]){
    // code section
    int dev_count;
    cudaGetDeviceCount(&dev_count);
    std::cout << "gpu count : " << dev_count << std::endl;

    cudaDeviceProp dev_prop;
    for (size_t i = 0; i < dev_count; i++)
    {
        cudaGetDeviceProperties(&dev_prop, i);
        std::cout << "device name : " << dev_prop.name << std::endl;
        std::cout << "SM count : " << dev_prop.multiProcessorCount << std::endl;
        std::cout << "max threads per block : " << dev_prop.maxThreadsPerBlock << std::endl;
        std::cout << "clock rate : " << dev_prop.clockRate << " kHz"<<std::endl;
        std::cout << "global mem : " <<dev_prop.totalGlobalMem << " B" <<std::endl;
        std::cout << "shared memory per block : "<<dev_prop.sharedMemPerBlock << " B" <<std::endl;

        for (size_t i = 0; i < 3; i++)
            std::cout << "max thread in " << (char)('x'+i) << " dim : " << dev_prop.maxThreadsDim[i] << std::endl;
        
        for (size_t i = 0; i < 3; i++)
            std::cout << "max block in " << (char)('x'+i) << " dim grid : " << dev_prop.maxGridSize[i] << std::endl;
        
    }
    
    return 0;
}