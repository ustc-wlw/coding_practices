
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define CHECK(call)                                                              \
{                                                                                \
    const cudaError_t error = call;                                              \
    if (error != cudaSuccess) {                                                  \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                            \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));      \
    }                                                                            \
}                                                                                \

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);

    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

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

void sumArraysOnHost(float *A, float *B, float *C, const int nx, const int ny)
{
    float *ia = A;
    float *ib = B;
    float *ic = C;

    for (int iy = 0; iy<ny; iy++){
        for (size_t ix = 0; ix < nx; ix++)
        {
            ic[ix] = ia[ix] + ib[ix];
        }
        ia += nx;
        ib += nx;
        ic += nx;
    }
}

__global__ void sumArraysOnDevice(float *A, float *B, float *C, const int nx, const int ny) {
    unsigned int ix = blockIdx.x * blockDim.x +  threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = iy * nx + ix;
    if (ix < nx && iy < ny) C[idx] = A[idx] + B[idx];
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

    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("device propertites: %s\n", deviceProp.name);
    CHECK(cudaSetDevice(dev));

    int nx = 1 << 14;
    int ny = 1 << 14;
    int n_Elem = nx * ny;
    printf("Matrix size %d, %d\n", nx, ny);
    size_t nBytes = n_Elem * sizeof(float);

    float *h_A, *h_B, *h_C, *gpuRef;
    h_A = (float*) malloc(nBytes);
    h_B = (float*) malloc(nBytes);
    h_C = (float*) malloc(nBytes);
    gpuRef = (float*) malloc(nBytes);

    double iStart, iElaps;

    printf("init host data start ....\n");
    iStart = cpuSecond();
    initData(h_A, n_Elem);
    initData(h_B, n_Elem);
    iElaps = cpuSecond() - iStart;
    printf("init cpu data time elaps: %5.2f\n", iElaps);

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

    int dimx = 32;
    int dimy = 16;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    iStart = cpuSecond();
    sumArraysOnDevice<<<grid, block>>> (d_A, d_B, d_C, nx, ny);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("sumArraysOnDevice <<<(%d, %d), (%d, %d)>>> time elaps: %f seconds\n", grid.x, grid.y, block.x, block.y, iElaps);

    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    printf("kernel result has copy to host\n");

    iStart = cpuSecond();
    sumArraysOnHost(h_A, h_B, h_C, nx, ny);
    iElaps = cpuSecond() - iStart;
    printf("sumArraysOnHost time elaps: %f second\n", iElaps);
    
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