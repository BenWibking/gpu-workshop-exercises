# Exercise 7: Performance-portable code

## Introduction

Here is another version of the code:

```
#include <cfloat>
#include <iostream>
#include <math.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

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

int main(void) {
  int N = 1e6;

  float *x_d;
  float *y_d;
#ifdef __CUDACC__
  cudaMalloc(&x_d, N * sizeof(float));
  cudaMalloc(&y_d, N * sizeof(float));
#else
  x_d = (float *)malloc(N * sizeof(float));
  y_d = (float *)malloc(N * sizeof(float));
#endif

  ParallelFor(N, [=] __host__ __device__(int i) {
    // this initializes the arrays
    x_d[i] = 1.0f;
    y_d[i] = 2.0f;
  });

  ParallelFor(N, [=] __host__ __device__(int i) {
    // this adds the two arrays
    y_d[i] = x_d[i] + y_d[i];
  });

  ParallelFor(N, [=] __host__ __device__(int i) {
    // this computes the error in y
    y_d[i] = fabs(y_d[i] - 3.0f);
  });

  // Find the maximum value of y_d
  thrust::device_ptr<float> d_ptr(y_d);
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
```

## Discussion

As a group, discuss the modifications to the code made since Exercise 6.

## Exercise A

Try to compile this code with:
```
$ nvcc -x cu add_gpu.cpp -o add_gpu
```

What error do you see? How can you fix the error?

Try to fix the error. Once you have compiled it successfully, you can run it with:
```
$ ./add_gpu
```

How fast is it?

## Exercise B

Now try to compile this code for CPU with:
```
$ gcc add_gpu.cpp -o add_cpu
```

Then, run it with:
```
$ ./add_cpu
```

How fast is it?

## Collective Discussion
