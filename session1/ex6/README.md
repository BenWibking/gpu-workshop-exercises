# Exercise 6: Adding two vectors (optimized)

## Introduction

Here is a GPU version of the code that avoids CPU-GPU data transfers (except for the final result):

```
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
