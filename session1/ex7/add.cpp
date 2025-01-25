#include <cfloat>
#include <iostream>
#include <math.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>

#ifdef __CUDACC__
template <typename T> __global__ void ParallelForKernelGPU(int N, T f) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < N; i += stride) {
    f(i);
  }
}
#endif

template <typename T> void ParallelForKernelCPU(int N, T f) {
  for (int i = 0; i < N; i++) {
    f(i);
  }
}

template <typename T> void ParallelFor(int N, T f) {
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
#ifdef __CUDACC__
  ParallelForKernelGPU<<<numBlocks, blockSize>>>(N, f);
#else
  ParallelForKernelCPU(N, f);
#endif
}

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

int main(void) {
  int N = 1e6;
  std::cout << "N = " << N << " elements\n";

  float *x_d;
  float *y_d;
#ifdef __CUDACC__
  cudaMalloc(&x_d, N * sizeof(float));
  cudaMalloc(&y_d, N * sizeof(float));
#else
  x_d = (float *)malloc(N * sizeof(float));
  y_d = (float *)malloc(N * sizeof(float));
#endif

  ParallelFor(N, [=] HOST_DEVICE(int i) {
    // this initializes the arrays
    x_d[i] = 1.0f;
    y_d[i] = 2.0f;
  });

  ParallelFor(N, [=] HOST_DEVICE(int i) {
    // this adds the two arrays
    y_d[i] = x_d[i] + y_d[i];
  });

  ParallelFor(N, [=] HOST_DEVICE(int i) {
    // this computes the error in y
    y_d[i] = fabs(y_d[i] - 3.0f);
  });

  // Find the maximum value of y_d
#ifdef __CUDACC__
  thrust::device_ptr<float> d_ptr(y_d);
#else
  float *d_ptr = y_d;
#endif

  auto max_iter = thrust::max_element(d_ptr, d_ptr + N);
  float maxError = *max_iter;
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
#ifdef __CUDACC__
  cudaFree(x_d);
  cudaFree(y_d);
#else
  free(x_d);
  free(y_d);
#endif
  return 0;
}
