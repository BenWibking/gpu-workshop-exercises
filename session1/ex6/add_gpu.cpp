#include <cfloat>
#include <iostream>
#include <math.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

__global__ void initialize(int n, float *x, float *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
}

__global__ void add(int n, float *x, float *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}

__global__ void computeError(int n, float *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    y[i] = fabs(y[i] - 3.0f);
  }
}

int main(void) {
  int N = 1e6;
  std::cout << "N = " << N << " elements\n";

  float *x_d;
  float *y_d;
  cudaMalloc(&x_d, N * sizeof(float));
  cudaMalloc(&y_d, N * sizeof(float));

  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  initialize<<<numBlocks, blockSize>>>(N, x_d, y_d);
  add<<<numBlocks, blockSize>>>(N, x_d, y_d);

  // Compute the error for each element and save it in y_d
  computeError<<<numBlocks, blockSize>>>(N, y_d);

  // Find the maximum value of y_d
  thrust::device_ptr<float> d_ptr(y_d);
  auto max_iter = thrust::max_element(d_ptr, d_ptr + N);
  float maxError = *max_iter;
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x_d);
  cudaFree(y_d);
  return 0;
}
