# Exercise 6: Adding two vectors (optimized)

## Introduction

Here is a GPU version of the code that avoids CPU-GPU data transfers (except for the final result):

```
#include <cfloat>
#include <iostream>
#include <math.h>

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

  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  initialize<<<numBlocks, blockSize>>>(N, x_d, y_d);
  add<<<numBlocks, blockSize>>>(N, x_d, y_d);

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
```

## Discussion

As a group, discuss the modifications to the code made since Exercise 5.

* Why are the additional GPU kernels necessary?

## Exercise

You can compile this code with:
```
$ nvcc -x cu add_gpu.cpp -o add_gpu
```

Now you can run it with:
```
$ ./add_gpu
```

What is the output?

## Timing comparison

Run the old version of the code and this new version and see how long each takes to run using the `time` command:
```
time ../ex5/add_gpu
```
and
```
time ./add_gpu
```

*Note:* The elapsed time (i.e., that you would measure on a stopwatch) is that reported as the number after "real:".

Which version is faster? By how much?

## Profiling

Now run this example with `nsys profile`:
```
$ nsys profile ./add_gpu
$ nsys stats report1.nsys-rep
```

Examine the output. How long did it take to run the kernel to add the two arrays with this version? How long does it take to run the kernel to initialize the arrays? How long does it take to run the kernels to compute the error?

## Collective Discussion

## Varying array size

Now, let's compare the CPU vs. GPU performance as a function of array size. We initially chose an array of 1e6 elements. Let's increase that by several orders of magnitude.

* Modify the code in `add_gpu.cpp` to run with 1e9 array elements and re-compile and re-run it. Write down the elapsed time output by `time ./add_gpu`.

* Modify the code in `../ex3/add_cpu.cpp` to run with 1e9 array elements and re-compile and re-run it. Write down the elapsed time output by `time ../ex3/add_cpu`.

How much faster is this version of the code compared to the CPU version of the code?

## Discussion

Why is this version faster than the version from the previous exercise (Exercise 5)?

## Collective Discussion
