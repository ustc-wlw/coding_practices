
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK(call)                                                              \
{                                                                                \
    const cudaError_t error = call;                                              \
    if (error != cudaSuccess) {                                                  \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                            \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));      \
    }                                                                            \
}                                                                                \

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

int main() {
    printf("main start ............\n");

    cudaSetDevice(0);

    int n_Elem = 32;
    size_t nBytes = n_Elem * sizeof(float);

    float *h_A, *h_B, *h_C, *gpuRef;
    h_A = (float*) malloc(nBytes);
    h_B = (float*) malloc(nBytes);
    h_C = (float*) malloc(nBytes);
    gpuRef = (float*) malloc(nBytes);

    printf("init host data start ....\n");
    initData(h_A, n_Elem);
    initData(h_B, n_Elem);

    memset(h_C, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    float *d_A, *d_B, *d_C;
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaMalloc((float**)&d_C, nBytes);


    printf("After init data and copy data from host to device start....\n");
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
    printf("copy data from host to device finished!!\n");

    dim3 block(32);
    dim3 grid((n_Elem + block.x - 1) / block.x);
    sumArraysOnDevice<<<grid, block>>> (d_A, d_B, d_C, n_Elem);
    printf("kernel function has lauched....\n");

    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    printf("kernel result has copy to host\n");

    sumArraysOnHost(h_A, h_B, h_C, n_Elem);
    
    printf("begain check result between host and device result...\n");
    checkResult(h_C, gpuRef, n_Elem);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);


    free(h_A);
    free(h_B);
    free(h_C);
    free(gpuRef);

}