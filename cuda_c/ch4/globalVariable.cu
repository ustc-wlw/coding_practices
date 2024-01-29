#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

/*
 * An example of using a statically declared global variable (devData) to store
 * a floating-point value on the device.
 */

__device__ float devData;

__global__ void checkGlobalVariable()
{
    printf("Device: original device global variabel is %f\n", devData);

    devData += 2.0;
}

int main()
{
    float value = 3.14;
    // CHECK(cudaMemcpyToSymbol(devData, &value, sizeof(float)));
    float *devPtr;
    CHECK(cudaGetSymbolAddress((void**)&devPtr, devData));
    CHECK(cudaMemcpy(devPtr, &value, sizeof(float), cudaMemcpyHostToDevice));
    printf("Host: copy value %f to device\n", value);

    checkGlobalVariable<<<1, 1>>> ();
    // CHECK(cudaDeviceSynchronize());

    // CHECK(cudaMemcpyFromSymbol(&value, devData, sizeof(float)));
    CHECK(cudaMemcpy(&value, devPtr, sizeof(float), cudaMemcpyDeviceToHost));
    printf("Host: after kenerl run, value is %f\n", value);

    CHECK(cudaDeviceReset());
    return 0;

}