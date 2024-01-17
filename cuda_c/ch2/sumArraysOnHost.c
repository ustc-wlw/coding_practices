#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cuda_runtime.h>

void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
    for (int i = 0; i<N; i++){
        C[i] = A[i] + B[i];
    }
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
    int n_Elem = 1024;
    size_t nBytes = n_Elem * sizeof(float);

    float *h_A, *h_B, *h_C;
    h_A = (float*) malloc(nBytes);
    h_B = (float*) malloc(nBytes);
    h_C = (float*) malloc(nBytes);

    // float *d_A, *d_B, *d_C;
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaMalloc((float**)&d_C, nBytes);

    initData(h_A, n_Elem);
    initData(h_B, n_Elem);

    sumArraysOnHost(h_A, h_B, h_C, n_Elem);

    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    free(h_A);
    free(h_B);
    free(h_C);

}