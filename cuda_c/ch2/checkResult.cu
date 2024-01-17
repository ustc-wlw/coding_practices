// #ifndef CHECKRESULT
// #define CHECKRESULT

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

// #endif