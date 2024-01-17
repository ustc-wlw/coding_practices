#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK(call)                                                              \
{                                                                                \
    const cudaError_t error = call;                                              \
    if (error != cudaSuccess) {                                                  \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                            \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));      \
    }                                                                            \
}                                                                                \

void initialInt(int *ip, int size) {
    for (size_t i = 0; i < size; i++)
    {
        ip[i] = i;
    }
}

void printMatrix(int *matrix, const int nx, const int ny) {
    int *ic = matrix;
    printf("\nMatrix: {%d, %d} \n", nx, ny);
    for (size_t y = 0; y < ny; y++)
    {
        for (size_t ix = 0; ix < nx; ix++)
        {
            printf("%d, ", ic[ix]);
        }
        ic += nx;
        printf("\n");
    }
    printf("\n");
}

__global__ void printThreadIndex(int *A, const int nx, const int ny) {
    int ix = blockIdx.x * blockDim.x  + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int idx = iy * nx + ix;
    printf("thread_id: (%d, %d), block_id: (%d, %d), coordinate: (%d, %d), "
    "global index: %2d, element value: %2d\n",
    threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, ix, iy, idx, A[idx]);

}

int main() {
    printf("main start ............\n");

    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("device propertites: %s\n", deviceProp.name);
    CHECK(cudaSetDevice(dev));

    int nx = 8;
    int ny = 6;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);

    int *h_A = (int*) malloc(nBytes);

    initialInt(h_A, nxy);
    printMatrix(h_A, nx, ny);

    int *d_A;
    cudaMalloc((void**)&d_A, nBytes);

    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);

    dim3 block(4, 2);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    printThreadIndex<<<block, grid>>> (d_A, nx, ny);

    cudaDeviceSynchronize();

    cudaFree(d_A);
    free(h_A);

    cudaDeviceReset();
}