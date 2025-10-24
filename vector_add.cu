#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float *A, const float *B, float *C, int n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        C[idx] = A[idx] + B[idx];
    }
}

int main()
{
    int n = 1000;
    size_t size = n * sizeof(float);

    float *host_a = (float *)malloc(size);
    float *host_b = (float *)malloc(size);
    float *host_c = (float *)malloc(size);

    for (int i = 0; i < n; i++)
    {
        host_a[i] = i;
        host_b[i] = 2 * i;
    }

    float *d_a;
    float *d_b;
    float *d_c;

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, host_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, host_b, size, cudaMemcpyHostToDevice);

    int threadPerBlock = 256;
    int numBlocks = (n + threadPerBlock - 1) / threadPerBlock;
    vectorAdd<<<numBlocks, threadPerBlock>>>(d_a, d_b, d_c, n);

    cudaMemcpy(host_c, d_c, size, cudaMemcpyDeviceToHost);

    // 8. 驗證結果
    for (int i = 0; i < 10; i++)
    {
        printf("c[%d] = %.1f (expected: %.1f)\n",
               i, host_c[i], host_a[i] + host_b[i]);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(host_a);
    free(host_b);
    free(host_c);
    return 0;
}