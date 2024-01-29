
#include "../common/common.h"
#include <stdio.h>
#include <cuda_runtime.h>

void checkResult(float *hostRef, float *gpuRef, const int N) {
    double epsilon = 1.0E-8;
    int match = 1;
    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("Array does not match\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
            break;
        }
        
    }
    if (match)
    {
        printf("Array match success!!\n");
    }
}

void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
    for (int i = 0; i<N; i++){
        C[i] = A[i] + B[i];
    }
}

__global__ void sumArraysOnDevice(float *A, float *B, float *C, const int N) {
    int tid = blockIdx.x * blockDim.x +  threadIdx.x;
    if (tid < N) C[tid] = A[tid] + B[tid];
}

void initData(float *ip, int size) {
    time_t t;
    srand((unsigned int) time(&t));

    for (size_t i = 0; i < size; i++)
    {
        ip[i] = (float) (rand() & 0xFF) / 10.0f;
    }
}

int main(int argc, char **argv) {
    printf("main start ............\n");

    int dev = 0;
    cudaSetDevice(dev);

    // get device properties
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));

    // check if support mapped memory
    if (!deviceProp.canMapHostMemory)
    {
        printf("Device %d does not support mapping CPU host memory!\n", dev);
        CHECK(cudaDeviceReset());
        exit(EXIT_SUCCESS);
    }

    printf("Using Device %d: %s ", dev, deviceProp.name);

    // set up data size of vectors
    int ipower = 10;

    if (argc > 1) ipower = atoi(argv[1]);

    int nElem = 1 << ipower;
    size_t nBytes = nElem * sizeof(float);

    if (ipower < 18)
    {
        printf("Vector size %d power %d  nbytes  %3.0f KB\n", nElem, ipower,
               (float)nBytes / (1024.0f));
    }
    else
    {
        printf("Vector size %d power %d  nbytes  %3.0f MB\n", nElem, ipower,
               (float)nBytes / (1024.0f * 1024.0f));
    }

    float *h_C, *gpuRef;
    h_C = (float*) malloc(nBytes);
    gpuRef = (float*) malloc(nBytes);

    float *ph_A, *ph_B, *d_C;
    cudaHostAlloc((void **)&ph_A, nBytes, cudaHostAllocMapped);
    cudaHostAlloc((void **)&ph_B, nBytes, cudaHostAllocMapped);
    cudaMalloc((void **)&d_C, nBytes);

    float *d_A, *d_B;
    cudaHostGetDevicePointer((void **)&d_A, (void *)ph_A, 0);
    cudaHostGetDevicePointer((void **)&d_B, (void *)ph_B, 0);

    printf("init host data start ....\n");
    initData(ph_A, nElem);
    initData(ph_B, nElem);

    memset(h_C, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // set up execution configuration
    int iLen = 512;
    dim3 block (iLen);
    dim3 grid((nElem + block.x - 1) / block.x);
    sumArraysOnDevice<<<grid, block>>> (d_A, d_B, d_C, nElem);

    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    printf("kernel result has copy to host\n");

    sumArraysOnHost(ph_A, ph_B, h_C, nElem);
    
    printf("begain check result between host and device result...\n");
    checkResult(h_C, gpuRef, nElem);

    cudaFreeHost(ph_A);
    cudaFreeHost(ph_B);
    cudaFree(d_C);

    free(h_C);
    free(gpuRef);

    // reset device
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}