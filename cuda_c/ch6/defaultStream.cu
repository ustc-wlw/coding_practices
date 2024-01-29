#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

__device__ void naiveCopy(int *in, int *o, const int N)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N)
    {
        o[idx] = in[idx];
    }
}

__global__ void kenerl1()
{
    printf("kernal 1 run .....\n");
    double sum = 0.0;
    for(int i = 0; i< 100000000; i++)
    {
        sum += i;
    }
    printf("kernal 1 finished !!\n");
}

__global__ void kenerl2()
{
    printf("kernal 2 run .....\n");
    double sum = 0.0;
    for(int i = 0; i< 100000000; i++)
    {
        sum += i;
    }
    printf("kernal 2 finished !!!\n");
}

__global__ void kenerl3()
{
    printf("kernal 3 run .....\n");
    double sum = 0.0;
    for(int i = 0; i< 100000000; i++)
    {
        sum += i;
    }
    printf("kernal 3 finished !!!!!\n");
}

int main()
{
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));


    cudaStream_t pStream1, pStream2;
    // CHECK(cudaStreamCreate(&pStream1));
    // CHECK(cudaStreamCreate(&pStream2));
    CHECK(cudaStreamCreateWithFlags(&pStream1, cudaStreamNonBlocking));
    CHECK(cudaStreamCreateWithFlags(&pStream2, cudaStreamNonBlocking));

    dim3 block(1);
    dim3 grid(1);

    kenerl1<<<grid, block, 0, pStream1>>> ();
    kenerl2<<<grid, block>>> ();
    kenerl3<<<grid, block, 0, pStream2>>> ();

    cudaDeviceSynchronize();

    // cudaStreamSynchronize(pStream);
    CHECK(cudaStreamDestroy(pStream1));
    CHECK(cudaStreamDestroy(pStream2));

    return 0;
}