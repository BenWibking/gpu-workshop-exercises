# Exercise 1

## Your first GPU kernel

We are going to start by showing how to run the simplest possible program on the GPU.

Here is the code:
```
__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

int main() {
    cuda_hello<<<1,1>>>(); 
    return 0;
}
```

Now you can compile it with:
```
$ nvcc hello.cpp -o hello
```
