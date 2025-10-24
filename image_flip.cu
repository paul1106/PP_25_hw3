#include <stdio.h>
#include <cuda_runtime.h>

__global__ void flipImage(unsigned char *input, unsigned char *output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int input_idx = y * width + x;
        int output_idx = (height - y - 1) * width + x;
        output[output_idx] = input[input_idx];
    }
}

int main()
{
    int width = 64;
    int height = 64;
    size_t size = width * height * sizeof(unsigned char);

    unsigned char *h_input = (unsigned char *)malloc(size);
    unsigned char *h_output = (unsigned char *)malloc(size);

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            h_input[i * width + j] = i * 4;
        }
    }

    unsigned char *d_input;
    unsigned char *d_output;

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize(4, 4);

    flipImage<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    printf("Original (first 3 rows):\n");
    for (int i = 0; i < 3; i++)
    {
        printf("Row %d: %d\n", i, h_input[i * width]);
    }

    printf("\nFlipped (first 3 rows):\n");
    for (int i = 0; i < 3; i++)
    {
        printf("Row %d: %d\n", i, h_output[i * width]);
    }

    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}