#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define BDIMX 32
#define BDIMY 32
#define IPAD 1

__global__ void setRowReadRow(int *out)
{
    __shared__ int tile[BDIMY][BDIMX];
    int id = threadIdx.y * blockDim.x + threadIdx.x;
    tile[threadIdx.y][threadIdx.x] = id;

    __syncthreads();

    out[id] = tile[threadIdx.y][threadIdx.x];
}

__global__ void setRowReadCol(int *out)
{
    __shared__ int tile[BDIMY][BDIMX];
    int id = threadIdx.y * blockDim.x + threadIdx.x;
    tile[threadIdx.y][threadIdx.x] = id;

    __syncthreads();

    out[id] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setRowReadColPad(int *out)
{
    __shared__ int tile[BDIMY][BDIMX + IPAD];
    int id = threadIdx.y * blockDim.x + threadIdx.x;
    tile[threadIdx.y][threadIdx.x] = id;

    __syncthreads();

    out[id] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setColReadCol(int *out)
{
    __shared__ int tile[BDIMX][BDIMY];
    int id = threadIdx.y * blockDim.x + threadIdx.x;
    tile[threadIdx.x][threadIdx.y] = id;

    __syncthreads();

    out[id] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setRowReadColDyn(int *out)
{
    extern __shared__ int tile[];
    unsigned int row_idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int col_idx = threadIdx.x * blockDim.y + threadIdx.y;

    tile[row_idx] = row_idx;

    __syncthreads();

    out[row_idx] = tile[col_idx];
}

int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    cudaSharedMemConfig pConfig;
    CHECK(cudaDeviceGetSharedMemConfig ( &pConfig ));
    printf("with Bank Mode:%s \n", pConfig == 1 ? "4-Byte" : "8-Byte");


    int nElm = BDIMX * BDIMY;
    size_t nSize = nElm * sizeof(int);

    int *out;
    cudaMalloc((void **)&out, nSize);
    cudaMemset((void *)out, 0, nElm);

    dim3 block(BDIMX, BDIMY);
    dim3 grid(1, 1);

    double iStart = seconds();
    setRowReadRow<<<block, grid>>> (out);
    CHECK(cudaDeviceSynchronize());
    double iElaps = seconds() - iStart;
    printf("setRowReadRow     <<< %4d, %4d >>> elapsed %f sec\n", grid.x,
           block.x, iElaps);
    CHECK(cudaGetLastError());

    iStart = seconds();
    setColReadCol<<<block, grid>>> (out);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("setColReadCol     <<< %4d, %4d >>> elapsed %f sec\n", grid.x,
           block.x, iElaps);
    CHECK(cudaGetLastError());

    iStart = seconds();
    setRowReadCol<<<block, grid>>> (out);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("setRowReadCol     <<< %4d, %4d >>> elapsed %f sec\n", grid.x,
           block.x, iElaps);
    CHECK(cudaGetLastError());

    iStart = seconds();
    setRowReadColPad<<<block, grid>>> (out);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("setRowReadColPad     <<< %4d, %4d >>> elapsed %f sec\n", grid.x,
           block.x, iElaps);
    CHECK(cudaGetLastError());

    iStart = seconds();
    setRowReadColDyn<<<block, grid, nSize>>> (out);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("setRowReadColDyn     <<< %4d, %4d >>> elapsed %f sec\n", grid.x,
           block.x, iElaps);
    CHECK(cudaGetLastError());

    CHECK(cudaFree(out));

    // reset device
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;

}