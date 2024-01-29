#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void kernal(int *in, int *o, const int N)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N)
    {
        o[idx] = in[idx];
    }
}

int main()
{
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    int nElem = 1 << 14;
    size_t nBytes = nElem * sizeof(int);

    dim3 block(32);
    dim3 grid(nElem / block.x);

    int *d_a;
    int *d_b;
    cudaMalloc((void**)&d_a, nBytes);
    cudaMemset((void *)d_a, 1, nElem);

    cudaMalloc((void**)&d_b, nBytes);
    cudaMemset((void *)d_b, 0, nElem);

    cudaStream_t pStream;
    CHECK(cudaStreamCreate(&pStream));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    cudaEventRecord(start, pStream);

    kernal<<<grid, block, 0, pStream>>>(d_a, d_b, nElem);

    cudaEventRecord(stop, pStream);

    cudaEventSynchronize(stop);

    float elapsed_time = 0.0;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("kernal elapsed time is %f\n", elapsed_time);

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    // cudaStreamSynchronize(pStream);
    CHECK(cudaStreamDestroy(pStream));
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));

    return 0;
}