#include <iostream>
#include <math.h>

__global__
void add(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}

__global__
void initialize(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
}

__global__
float verify(int n, float *y);
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  
  float maxError = 0.0f;
  for (int i = index; i < n; i += stride) {
    // TODO: check y[i]
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  }
  return maxError;
}

int main(void)
{
  int N = 1e6;

  float *x_d;
  float *y_d;
  cudaMalloc(&x_d, N*sizeof(float));
  cudaMalloc(&y_d, N*sizeof(float));

  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  initialize<<<numBlocks, blockSize>>>(N, x_d, y_d);
  add<<<numBlocks, blockSize>>>(N, x_d, y_d);
  float maxError = verify<<<numBlocks, blockSize>>>(N, y_d);

  cudaDeviceSynchronize();

  cudaFree(x_d);
  cudaFree(y_d);

  std::cout << "Max error: " << maxError << std::endl;
  return 0;
}
