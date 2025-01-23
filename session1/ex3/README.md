# Exercise 3: Adding two vectors

## Introduction

Here is a CPU code to add two vectors of 1 million elements each:
```
#include <iostream>
#include <math.h>
#include <vector>

// function to add the elements of two arrays
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
      y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 1e6; // 1 million elements

  std::vector<float> x(N);
  std::vector<float> y(N);

  // initialize x and y arrays
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the CPU
  add(N, x.data(), y.data());

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++) {
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  }
  std::cout << "Max error: " << maxError << std::endl;

  return 0;
}
```

## Discussion

As a group, discuss each line of code and hypothesize what it does and why it's included in this example code.

## Exercise

You can compile this code with:
```
$ nvcc -x cu add_cpu.cpp -o add_cpu
```

Now you can run it with:
```
$ ./add_cpu
```

What is the output?

## GPU version

Here is a GPU version of the same code:

```
#include <iostream>
#include <math.h>
#include <vector>

__global__
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
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

  add<<<1, 1>>>(N, x_d, y_d);

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

As a group, discuss each new line of code that has been added to the GPU version of the code and hypothesize what it does and why it's needed to run successfully on the GPU.

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

## Profiling

Now run this example with `nsys profile`:
```
$ nsys profile ./add_gpu
$ nsys stats report1.nsys-rep
```

Examine the output. How long did it take to add the two arrays? How long did it take to transfer the input data to the GPU? How long did it take to transfer the result back to the CPU? Provide your answers in milliseconds.

Examine the output again. How much data (in megabytes) was transferred from the CPU to the GPU? How much data (in megabytes) was transferred from the GPU to the CPU? Using these numbers, calculate the bandwidth (in gigabytes per second) of data transfer between CPU and GPU and vice versa.

## Timing comparison

Run each version of the code and see how long each takes to run using the `time` command:
```
time ./add_cpu
```
and
```
time ./add_gpu
```

*Note:* The elapsed time (i.e., that you would measure on a stopwatch) is that reported as the number after "real:".

Which version is faster? By how much?

## Discussion

You should have found that the *CPU* version is substantially faster. Discuss as a group and form a hypothesis as to why this is the case for this example.

## Collective discussion

We will discuss the results of each group. I will provide additional context.
