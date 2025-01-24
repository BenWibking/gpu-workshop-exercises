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

__global__ void reduceMax(float *data, float *result, int n) {
  extern __shared__ float shared[]; // Dynamic shared memory

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Load elements into shared memory
  if (i < n) {
    shared[tid] = data[i];
  } else {
    shared[tid] = -FLT_MAX; // Use minimum float value for out-of-bounds threads
  }
  __syncthreads();

  // Perform reduction in shared memory
  for (int s = blockDim.x / 2; s > 0; s /= 2) {
    if (tid < s) {
      shared[tid] = fmaxf(shared[tid], shared[tid + s]);
    }
    __syncthreads();
  }

  // Write the block's result to the output array
  if (tid == 0) {
    result[blockIdx.x] = shared[0];
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

  // The remainder of this code is verifies that the above calculation was
  // performed correctly.
  computeError<<<numBlocks, blockSize>>>(N, y_d);

  // 1. Compute the maximum error in y_d within each thread block
  float *partialMax_d;
  cudaMalloc(&partialMax_d, numBlocks * sizeof(float));
  int sharedMemSize = blockSize * sizeof(float);
  reduceMax<<<numBlocks, blockSize, sharedMemSize>>>(y_d, partialMax_d, N);

  // 2. Iteratively reduce the partial results if numBlocks > 256
  float *d_input = partialMax_d;
  float *d_output;
  cudaMalloc(&d_output, numBlocks * sizeof(float));

  int remainingBlocks = numBlocks;
  while (remainingBlocks > 1) {
    int newNumBlocks = (remainingBlocks + blockSize - 1) / blockSize;
    // Kernel uses d_input as input and d_output as output
    reduceMax<<<newNumBlocks, blockSize, sharedMemSize>>>(d_input, d_output,
                                                          remainingBlocks);
    remainingBlocks = newNumBlocks;

    // Swap input and output for the next iteration
    std::swap(d_input, d_output);
  }

  // Copy the final result back to the host
  float maxError;
  cudaMemcpy(&maxError, d_input, sizeof(float), cudaMemcpyDeviceToHost);
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x_d);
  cudaFree(y_d);
  cudaFree(partialMax_d);
  cudaFree(d_output);
  return 0;
}
