#include <stdio.h>
#include <stdlib.h>
#include<cuda.h>
#include <math.h>

#define THREADS_PER_BLOCK 16
#define TILE_WIDTH 2
__global__ void MatrixMulKernel(float* d_M, float* d_N, float* d_P, int Width)
{
	 __shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
	 __shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];
	 int bx = blockIdx.x; int by = blockIdx.y;
	 int tx = threadIdx.x; int ty = threadIdx.y;
	// Identify the row and column of the Pd element to work on
	 int Row = by * TILE_WIDTH + ty;
	 int Col = bx * TILE_WIDTH + tx;
	 float Pvalue = 0;
	// Loop over the Md and Nd tiles required to compute the Pd element
	 for (int m = 0; m < Width / TILE_WIDTH; ++m) {
		 // Coolaborative loading of Md and Nd tiles into shared memory
		 ds_M[ty][tx] = d_M[Row*Width + m * TILE_WIDTH + tx];
		 ds_N[ty][tx] = d_N[(m*TILE_WIDTH + ty)*Width + Col];
		 __syncthreads();
		 for (int k = 0; k < TILE_WIDTH; ++k)
			 Pvalue += ds_M[ty][k] * ds_N[k][tx];
		 __syncthreads();
	}
 // d_P[Row*Width + Col] = Pvalue;
}
void cpu_matrixmult(int *a, int *b, int *c, int n)
{
	int sum, Row, Col, k;
	for (int Row = 0; Row < n; Row++)
	{
		for (int Col = 0; Col < n; Col++)
		{
			sum = 0;
			for (k = 0; k < n; k++)
			{
				sum += a[Row*n + k] * b[n*k + Col];
			}
			c[Row*n + Col] = sum;


		}
	}
}
__global__ void gpu_matrixmult(int *a, int *b, int *c, int n)
{
	int k, sum = 0;
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	  if (col < n && row < n)
	  { 
		  for (k = 0; k < n; k++) 
			  sum += a[row * n + k] * b[k * n + col]; 
		  c[row * n + col] = sum; 
	  } 
}


int main(int argc, char *argv[])
{
	int *a, *b, *c, *d, *e;
	int *dev_a, *dev_b, *dev_d, *dev_e;
	cudaEvent_t start, stop;
	int n = 1200;
	int Grid_Dim_x = (n/ THREADS_PER_BLOCK)+1, Grid_Dim_y = (n / THREADS_PER_BLOCK) + 1;
	int Block_Dim_x = THREADS_PER_BLOCK, Block_Dim_y = THREADS_PER_BLOCK;
	int size = n * n * sizeof(int);
	a = (int*)malloc(size);
	b = (int*)malloc(size);
	c = (int*)malloc(size);
	d = (int*)malloc(size);
	e = (int*)malloc(size);
	float elapsed_time_ms;
	dim3 Grid(Grid_Dim_x, Grid_Dim_y,1);
	dim3 Block(Block_Dim_x, Block_Dim_y, 1);
	for (int i = 0; i < n*n; i++)
	{
		a[i] = 2;
		b[i] = 2;
	}
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	cpu_matrixmult(a, b, c, n);
	cudaEventRecord(stop, 0); // measure end time
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time_ms, start, stop);
	printf("Time to calculate results on CPU: %f ms.\n", elapsed_time_ms);
	cudaEventRecord(start, 0);
	cudaMalloc((void**)&dev_a, size); // allocate memory on device
	cudaMalloc((void**)&dev_b, size);
	cudaMalloc((void**)&dev_d, size);
	cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
	gpu_matrixmult << <Grid, Block >> > (dev_a, dev_b, dev_d, n);
	cudaMemcpy(d, dev_d, size, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0); // measure end time
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time_ms, start, stop);
	printf("Time to calculate results on GPU: %f ms.\n", elapsed_time_ms);
	cudaEventRecord(start, 0);
	cudaMalloc((void**)&dev_a, size); // allocate memory on device
	cudaMalloc((void**)&dev_b, size);
	cudaMalloc((void**)&dev_e, size);
	cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
	gpu_matrixmult << <Grid, Block >> > (dev_a, dev_b, dev_e, n);
	cudaMemcpy(e, dev_e, size, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0); // measure end time
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time_ms, start, stop);
	printf("Time to calculate results on tiledMatrixKernelsn %f ms.\n", elapsed_time_ms);
	return 0;
}

