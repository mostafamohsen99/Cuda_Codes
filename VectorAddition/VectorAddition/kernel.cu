#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#define N (2048*2048)
#define THREADS_PER_BLOCK 512


__global__ void vecadd(int *a, int *b, int *c, int n)
{
	// get our global thread id
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	// make sure we do not go out of bounds
	if (id < n)
		c[id] = a[id] + b[id];
}


int main(int argc, char* argv[])
{
	int *h_a, *h_b, *h_c;
	int *d_a, *d_b, *d_c;
	int size = N * sizeof(int);
	h_a = (int*)malloc(size);
	h_b = (int*)malloc(size);
	h_c = (int*)malloc(size);
	cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_b, size);
	cudaMalloc((void**)&d_c, size);
	int i;
	for (i = 0; i < N; i++)
	{
		h_a[i] = 1;
		h_b[i] = 1;
	}
	cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
	vecadd << < N / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (d_a, d_b, d_c,N);
	cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
	int sum = 0;
	for (i = 0; i < N; i++)
	{
		sum += h_c[i];
	}
	printf("final result: %d\n", sum / N);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	// release host memory
	free(h_a);
	free(h_b);
	free(h_c);

	return 0;

}
