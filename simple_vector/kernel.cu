
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void add_vector(int *a, int *b, int *c)
{
	int index = threadIdx.x;
	c[index] = a[index] + b[index];
	return;
}

int main()
{

	return;
}