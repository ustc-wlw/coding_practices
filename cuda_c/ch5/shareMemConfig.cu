#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

int main()
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    cudaSharedMemConfig config;
    CHECK(cudaDeviceGetSharedMemConfig(&config));
    printf("cudaSharedMemConfig is %d\n", config);

    /*
    enum __device_builtin__ cudaSharedMemConfig
    {
        cudaSharedMemBankSizeDefault   = 0,
        cudaSharedMemBankSizeFourByte  = 1,
        cudaSharedMemBankSizeEightByte = 2
    };
    */

   cudaFuncCache cacheConfig;
   cudaDeviceSetCacheConfig(cacheConfig);
   /*
    enum __device_builtin__ cudaFuncCache
    {
        cudaFuncCachePreferNone   = 0,    // < Default function cache configuration, no preference
        cudaFuncCachePreferShared = 1,    //**< Prefer larger shared memory and smaller L1 cache
        cudaFuncCachePreferL1     = 2,    //**< Prefer larger L1 cache and smaller shared memory
        cudaFuncCachePreferEqual  = 3     //**< Prefer equal size L1 cache and shared memory
    };
   */

    return 0;
}