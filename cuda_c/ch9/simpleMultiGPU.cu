#include "../common/common.h"
#include <stdio.h>
#include <cuda_runtime.h>

// nvprof --print-gpu-trace ./simpleMultiGPU 2

#define BDIM 512

void initialData(float *ip, int size)
{
    int i;

    for(i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
    for (int idx = 0; idx < N; idx++)
        C[idx] = A[idx] + B[idx];
}

void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if(i == 0) printf("hostRef first element %f, gpuRef first element %f\n", hostRef[i], gpuRef[i]);
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at %d\n", hostRef[i], gpuRef[i], i);
            break;
        }
    }

    if (match) printf("Arrays match.\n\n");
}

__global__ void iKernel(float *da, float *db, float *dc, const int nElem)
{
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < nElem)
    {
        dc[idx] = da[idx] + db[idx];
    }
    
}

int main(int argc, char **argv)
{
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("> %s Starting...\n", argv[0]);
    printf("> Using Device %d: %s\n", dev, deviceProp.name);
    
    int ngpus;
    CHECK(cudaGetDeviceCount(&ngpus));
    printf(" CUDA capable devices: %i\n", ngpus);

    int NGPUS = ngpus;
    if(argc > 1)
    {
        NGPUS = atoi(argv[1]);
    }

    float *d_A[NGPUS], *d_B[NGPUS], *d_C[NGPUS];
    float *h_A[NGPUS], *h_B[NGPUS], *hostRef[NGPUS], *gpuRef[NGPUS];
    cudaStream_t streams[NGPUS];

    int nElem = 1 << 24;
    int iSize = nElem / NGPUS;
    int nBytes = iSize * sizeof(float);
    printf("total array size is %dM, using %d devices with each device handing %dM\n", (nElem / 1024 / 1024), NGPUS, (iSize / 1024 / 1024));

    for(int i = 0; i<NGPUS; i++)
    {
        CHECK(cudaSetDevice(i));
        // allocate page locked host memory for asynchronous data transfer
        cudaMallocHost((void **)&h_A[i], nBytes);
        cudaMallocHost((void **)&h_B[i], nBytes);
        cudaMallocHost((void **)&hostRef[i], nBytes);
        cudaMallocHost((void **)&gpuRef[i], nBytes);

        cudaMalloc((void **)&d_A[i], nBytes);
        cudaMalloc((void **)&d_B[i], nBytes);
        cudaMalloc((void **)&d_C[i], nBytes);

        cudaStreamCreate(&streams[i]);
    }

    for(int i = 0; i< NGPUS; i++)
    {
        cudaSetDevice(i);
        initialData(h_A[i], iSize);
        initialData(h_B[i], iSize);
    }

    dim3 block(BDIM);
    dim3 grid(iSize / block.x);
    printf("Block dim %d, grid dim %d\n", block.x, grid.x);

    // record start time
    double iStart = seconds();

    for (size_t i = 0; i < NGPUS; i++)
    {
        CHECK(cudaSetDevice(i));
        CHECK(cudaMemcpyAsync(d_A[i], h_A[i], nBytes, cudaMemcpyHostToDevice, streams[i]));
        CHECK(cudaMemcpyAsync(d_B[i], h_B[i], nBytes, cudaMemcpyHostToDevice, streams[i]));
        iKernel<<<grid, block, 0, streams[i]>>> (d_A[i], d_B[i], d_C[i], iSize);

        CHECK(cudaMemcpyAsync(gpuRef[i], d_C[i], nBytes, cudaMemcpyDeviceToHost, streams[i]));
    }

    for (size_t i = 0; i < NGPUS; i++)
    {
        CHECK(cudaSetDevice(i));
        CHECK(cudaStreamSynchronize(streams[i]));
    }

    // calculate the elapsed time in seconds
    double iElaps = seconds() - iStart;
    printf("%d GPU timer elapsed: %8.2fms \n", NGPUS, iElaps * 1000.0);

    for (size_t i = 0; i < NGPUS; i++)
    {
        sumArraysOnHost(h_A[i], h_B[i], hostRef[i], iSize);
        checkResult(hostRef[i], gpuRef[i], iSize);
    }

    for (size_t i = 0; i < NGPUS; i++)
    {
        CHECK(cudaSetDevice(i));
        CHECK(cudaFree(d_A[i]));
        CHECK(cudaFree(d_B[i]));
        CHECK(cudaFree(d_C[i]));

        CHECK(cudaStreamDestroy(streams[i]));

        CHECK(cudaFreeHost(h_A[i]));
        CHECK(cudaFreeHost(h_B[i]));
        CHECK(cudaFreeHost(hostRef[i]));
        CHECK(cudaFreeHost(gpuRef[i]));

        CHECK(cudaDeviceReset());
    }
}