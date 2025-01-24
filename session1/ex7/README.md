# Exercise 7: Adding two vectors (with C++ lambda functions)

## Introduction

Here is a GPU version of the code that uses C++ lambda functions:

```
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
```

## Discussion

As a group, discuss the modifications to the code made since Exercise 6.

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
time ../ex6/add_gpu
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

Examine the output. How does it differ from the previous example?

## Collective Discussion
