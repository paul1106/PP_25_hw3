// hw3.cu - Stage 1: Basic Framework
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// GLM vector math library
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

// lodepng for PNG I/O
#include "lodepng.h"

// Type definitions (consistent with CPU version)
typedef glm::dvec2 vec2;
typedef glm::dvec3 vec3;

// Image parameters
unsigned int width;
unsigned int height;

// Camera parameters
vec3 camera_pos;
vec3 target_pos;

// Stage 1: Simple kernel - fill with black color for testing
__global__ void renderKernel(unsigned char *image, int width, int height)
{
    // Calculate 2D coordinates
    int j = blockIdx.x * blockDim.x + threadIdx.x; // x (column)
    int i = blockIdx.y * blockDim.y + threadIdx.y; // y (row)

    // Boundary check
    if (i >= height || j >= width)
        return;

    // Calculate pixel index (RGBA, 4 bytes per pixel)
    int idx = (i * width + j) * 4;

    // Fill with black for testing
    image[idx + 0] = 0;   // R
    image[idx + 1] = 0;   // G
    image[idx + 2] = 0;   // B
    image[idx + 3] = 255; // A (opaque)
}

int main(int argc, char **argv)
{
    // Step 1: Check argument count
    if (argc != 10)
    {
        printf("Usage: %s x1 y1 z1 x2 y2 z2 width height filename\n", argv[0]);
        return 1;
    }

    // Step 2: Parse command line arguments
    camera_pos = vec3(atof(argv[1]), atof(argv[2]), atof(argv[3]));
    target_pos = vec3(atof(argv[4]), atof(argv[5]), atof(argv[6]));
    width = atoi(argv[7]);
    height = atoi(argv[8]);
    const char *filename = argv[9];

    printf("Camera: (%.3f, %.3f, %.3f)\n", camera_pos.x, camera_pos.y, camera_pos.z);
    printf("Target: (%.3f, %.3f, %.3f)\n", target_pos.x, target_pos.y, target_pos.z);
    printf("Image size: %d x %d\n", width, height);

    // Step 3: Allocate host memory
    size_t image_size = width * height * 4 * sizeof(unsigned char); // RGBA
    unsigned char *h_image = (unsigned char *)malloc(image_size);

    // Step 4: Allocate device memory
    unsigned char *d_image;
    cudaMalloc(&d_image, image_size);

    // Step 5: Configure 2D grid and block dimensions
    dim3 blockSize(16, 16); // 256 threads per block
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y);

    printf("Launching kernel: grid(%d, %d), block(%d, %d)\n",
           gridSize.x, gridSize.y, blockSize.x, blockSize.y);

    // Step 6: Launch kernel
    renderKernel<<<gridSize, blockSize>>>(d_image, width, height);

    // Synchronize and check for errors
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Step 7: Copy results back to CPU
    cudaMemcpy(h_image, d_image, image_size, cudaMemcpyDeviceToHost);

    // Step 8: Save PNG file
    unsigned error = lodepng_encode32_file(filename, h_image, width, height);
    if (error)
    {
        printf("PNG error %u: %s\n", error, lodepng_error_text(error));
        return 1;
    }

    printf("Image saved to %s\n", filename);

    // Step 9: Free memory
    free(h_image);
    cudaFree(d_image);

    return 0;
}