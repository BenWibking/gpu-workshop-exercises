# Exercise 5: Adding two vectors, x2

## Introduction

Here is a GPU version of the code that launches a kernel with multiple thread blocks and 256 threads per thread block:

```
#include <iostream>
#include <math.h>
#include <vector>

__global__
void add(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 1e6;

  std::vector<float> x(N);
  std::vector<float> y(N);

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  float *x_d;
  float *y_d;
  cudaMalloc(&x_d, N*sizeof(float));
  cudaMalloc(&y_d, N*sizeof(float));

  cudaMemcpy(x_d, x.data(), N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(y_d, y.data(), N*sizeof(float), cudaMemcpyHostToDevice);

  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  add<<<numBlocks, blockSize>>>(N, x_d, y_d);

  cudaDeviceSynchronize();

  cudaMemcpy(y.data(), y_d, N*sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(x_d);
  cudaFree(y_d);

  float maxError = 0.0f;
  for (int i = 0; i < N; i++) {
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  }
  std::cout << "Max error: " << maxError << std::endl;

  return 0;
}
```

## Discussion

As a group, discuss the modifications to the code made since Exercise 4.

* How many thread blocks will this code launch?

* Why are the modifications to the `add` function necessary?

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
time ../ex4/add_gpu
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

Examine the output. How long did it take to run the kernel to add the two arrays with this version? How does this time compare with the time it takes to transfer the arrays from CPU to GPU?

## Collective Discussion

## Varying array size

Now, let's compare the CPU vs. GPU performance as a function of array size. We initially chose an array of 1e6 elements. Let's increase that by several orders of magnitude. We will find that for large enough arrays, adding two arrays is faster on the GPU.

* Modify the code in `add_gpu.cpp` to run with 1e7 array elements and re-compile and re-run it. Write down the elapsed time output by `time ./add_gpu`.

* Modify the code in `../ex3/add_cpu.cpp` to run with 1e7 array elements and re-compile and re-run it. Write down the elapsed time output by `time ../ex3/add_cpu`.

* Repeat the above two steps for N = 1e8 array elements.

* Repeat the above for N = 1e9 array elements.

NOTE: It may be easier to modify the GPU code and run it for all array sizes first, and then modify the CPU code in `../ex3` and run it for all array sizes.

At what size does running on the GPU become faster than running on the CPU?

## Discussion

Could you improve the performance of the GPU code?

What factors could change the threshold array size for which the GPU is faster?

## Collective Discussion
