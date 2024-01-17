#include <stdio.h>

__global__ void helloFromGPU()
{
    printf("Hello from GPU from thread id %d!\n", threadIdx.x);
}

int main()
{
    printf("Hello World from CPU!\n");

    helloFromGPU<<<1, 10>>>();
    // cudaDeviceReset();
    cudaDeviceSynchronize();
    return 0;
}
