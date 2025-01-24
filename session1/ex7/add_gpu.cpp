#include <cfloat>
#include <iostream>
#include <math.h>

template <typename T> __global__ void ParallelForKernel(int n, T f) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    f(i);
  }
}

template <typename T> void ParallelFor(int n, T f) {
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  ParallelForKernel<<<numBlocks, blockSize>>>(N, f);
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

  float *x_d;
  float *y_d;
  cudaMalloc(&x_d, N * sizeof(float));
  cudaMalloc(&y_d, N * sizeof(float));

  ParallelFor(
      N, __device__[=](int i) {
        // this initializes the arrays
        x_d[i] = 1.0f;
        y_d[i] = 2.0f;
      });

  ParallelFor(
      N, __device__[=](int i) {
        // this adds the two arrays
        y_d[i] = x_d[i] + y_d[i];
      });

  ParallelFor(
      N, __device__[=](int i) {
        // this computes the error in y
        y_d[i] = fabs(y_d[i] - 3.0f);
      });

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
