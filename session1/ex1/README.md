# Exercise 1: Your first GPU code

## Introduction

We are going to start by showing how to run the simplest possible program on the GPU.

Here is the code:
```
#include <cstdio>

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

int main() {
    cuda_hello<<<1,1>>>(); 
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

What happened?

Try running it with the NVIDIA Nsight Systems profiler:
```
$ nsys profile ./hello
```

What happened now?

Let's look at the profiling data:
```
$ nsys stats report1.nsys-rep
```

What do you see?

## Discussion

As a group, determine how long it took the `cuda_hello()` kernel to run based on the output from this command.

As a group, make a hypothesis as to why you get different output when running `./hello` versus running `nsys profile ./hello`.

## Collective discussion

We will discuss each group's hypotheses and I will provide some additional context.

*NOTE: Please do NOT yet go on to exercise 2. It may spoil the discussion for other groups.*
