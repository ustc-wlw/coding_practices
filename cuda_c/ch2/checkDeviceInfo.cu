#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    printf("device count is %d\n", deviceCount);

    int dev, driverVersion = 0, runtimeVersion = 0;
    dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("device name is %s\n", prop.name);

    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("driver version: %d, runtime version: %d\n", driverVersion, runtimeVersion);

    printf("cuda capability Major/Mnior version: %d.%d\n", prop.major, prop.minor);

    printf("HBM size: %.2f MB\n", (float)prop.totalGlobalMem / (pow(1024.0, 3)));

    printf("Bus bindwith %d bit\n", prop.memoryBusWidth);

    if (prop.l2CacheSize)
    {
        printf("l2CacheSize %d bytes\n", prop.l2CacheSize);
    }

    printf("Total of shared memory per block %lu bytes\n", prop.sharedMemPerBlock);
    printf("Total of registers per block %d\n", prop.regsPerBlock);
    printf("Total of shared memory per SM %lu bytes\n", prop.sharedMemPerMultiprocessor);

    printf("wrap size: %d\n", prop.warpSize);
    printf("SM number: %d\n", prop.multiProcessorCount);
    printf("maximum blocks per SM:  %d\n", prop.maxBlocksPerMultiProcessor);
    printf("maximum threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("maximum threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("maximum block dim: %d, %d, %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("maximum grid dim: %d, %d, %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);


}

/*
device count is 8
device name is Tesla V100-SXM2-32GB
driver version: 12020, runtime version: 11070
cuda capability Major/Mnior version: 7.0
HBM size: 31.74 MB
Bus bindwith 4096 bit
l2CacheSize 6291456 bytes
Total of shared memory per block 49152 bytes
Total of registers per block 65536
Total of shared memory per SM 98304 bytes
wrap size: 32
SM number: 80
maximum blocks per SM:  32
maximum threads per SM: 2048
maximum threads per block: 1024
maximum block dim: 1024, 1024, 64
maximum grid dim: 2147483647, 65535, 65535
*/