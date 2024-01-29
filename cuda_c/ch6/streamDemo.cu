#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

int main()
{
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    int leastPriority;
    int greatestPriority;
    CHECK(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
    printf("least Priority is %d, greatest Priority is %d\n", leastPriority, greatestPriority);

    cudaStream_t pStream;
    CHECK(cudaStreamCreate(&pStream));
    CHECK(cudaStreamDestroy(pStream));

    cudaEvent_t pEvent;
    CHECK(cudaEventCreate(&pEvent));
    CHECK(cudaEventDestroy(pEvent));



    return 0;
}