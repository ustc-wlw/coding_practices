#include "../common/common.h"
#include <stdio.h>
#include <cuda_runtime.h>

inline void enableP2P(const int ngpus)
{
    for (int i = 0; i < ngpus; i++)
    {
        cudaSetDevice(i);
        for (int j = 0; j < ngpus; j++)
        {
            if (i == j) continue;
            int peer_device_available = 0;
            cudaDeviceCanAccessPeer(&peer_device_available, i, j);
            if (peer_device_available)
            {
                cudaDeviceEnablePeerAccess(j, 0);
                printf("GPU%d,  can access peer memory of GPU%d \n", i, j);
            }
            else
            {
                printf("(GPU%d, GPU%d) can not access peer memory\n", i, j);
            }
        }
    }
}

inline void disableP2P (int ngpus)
{
    for( int i = 0; i < ngpus; i++ )
    {
        CHECK(cudaSetDevice(i));

        for(int j = 0; j < ngpus; j++)
        {
            if( i == j ) continue;

            int peer_access_available = 0;
            CHECK(cudaDeviceCanAccessPeer( &peer_access_available, i, j) );

            if( peer_access_available )
            {
                CHECK(cudaDeviceDisablePeerAccess(j));
                printf("> GPU%d disabled direct access to GPU%d\n", i, j);
            }
        }
    }
}

int main(int args, char **argv)
{
    int ngpus;
    CHECK(cudaGetDeviceCount(&ngpus));
    printf(" CUDA capable devices: %i\n", ngpus);

    // enableP2P(ngpus);

    int nElem = 1 << 10;
    int nBytes = nElem * sizeof(float);

    float *d_mem[2];
    for (size_t i = 0; i < 2; i++)
    {
        CHECK(cudaSetDevice(i));
        CHECK(cudaMalloc((void **)&d_mem[i], nBytes));
        CHECK(cudaMemset(d_mem[i], i, nElem));
    }
    
    CHECK(cudaSetDevice(0));
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    cudaEventRecord(start);
    for (size_t i = 0; i < 100; i++)
    {
        if (i % 2 == 0)
        {
            CHECK(cudaMemcpy(d_mem[1], d_mem[0], nBytes, cudaMemcpyDeviceToDevice));
        } else {
            CHECK(cudaMemcpy(d_mem[0], d_mem[1],  nBytes, cudaMemcpyDeviceToDevice));
        }
    }
    
    CHECK(cudaSetDevice(0));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_time_ms;
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);

    elapsed_time_ms /= 100.0f;
    printf("Ping-pong unidirectional cudaMemcpy:\t\t %8.2f ms ",
           elapsed_time_ms);
    printf("performance: %8.2f GB/s\n",
            (float)nBytes / (elapsed_time_ms * 1e6f));

    for (size_t i = 0; i < 2; i++)
    {
        CHECK(cudaSetDevice(i));
        CHECK(cudaFree(d_mem[i]));
    }

    CHECK(cudaSetDevice(0));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

}

