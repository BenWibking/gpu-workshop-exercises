# Exercise 2: Your first GPU code, fixed

## Introduction

I am going to provide a solution to the problem experienced in exercise 1.

Here is the code:
```
#include <cstdio>

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

int main() {
    cuda_hello<<<1,1>>>();
    
    cudaDeviceSynchronize();
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    
    return 0;
}
```

## Discussion

As a group, discuss each line of code and hypothesize what it does and why it's included in this example code.

## Exercise

You can compile this code with:
```
$ nvcc -x cu hello.cpp -o hello
```

Now you can run it with:
```
$ ./hello
```

What happened this time?

## Discussion

As a group, discuss why this code solves the problem experienced in exercise 1.

## Collective discussion

We will discuss each group's hypotheses and I will provide some additional context.
