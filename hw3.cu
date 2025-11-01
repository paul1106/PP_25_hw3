// hw3.cu - Stage 1: Basic Framework
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// GLM vector math library
// #define GLM_COMPILER 0
#define GLM_FORCE_CUDA
#define CUDA_VERSION 12800 // CUDA 12.8
#include <glm/glm.hpp>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <lodepng.h>

#define GLM_FORCE_SWIZZLE // vec3.xyz(), vec3.xyx() ...ect, these are called "Swizzle".
// https://glm.g-truc.net/0.9.1/api/a00002.html
//
#include <glm/glm.hpp>
// for the usage of glm functions
// please refer to the document: http://glm.g-truc.net/0.9.9/api/a00143.html
// or you can search on google with typing "glsl xxx"
// xxx is function name (eg. glsl clamp, glsl smoothstep)

#define pi 3.1415926535897932384626433832795f

typedef glm::vec2 vec2; // float precision 2D vector (x, y) or (u, v)
typedef glm::vec3 vec3; // 3D vector (x, y, z) or (r, g, b)
typedef glm::vec4 vec4; // 4D vector (x, y, z, w)
typedef glm::mat3 mat3; // 3x3 matrix

// Host variables
unsigned int width;          // image width
unsigned int height;         // image height
unsigned char *h_raw_image;  // 1D image
unsigned char **h_image_ptr; // 2D image

// Device constant memory (GPU can access, read-only, fast)
__constant__ int d_AA;
__constant__ float d_power;
__constant__ float d_md_iter;
__constant__ float d_ray_step;
__constant__ float d_shadow_step;
__constant__ float d_step_limiter;
__constant__ float d_ray_multiplier;
__constant__ float d_bailout;
__constant__ float d_eps;
__constant__ float d_FOV;
__constant__ float d_far_plane;
__constant__ vec3 d_camera_pos;
__constant__ vec3 d_target_pos;
__constant__ vec2 d_iResolution;

// mandelbulb distance function (DE)
// v = v^8 + c
// p: current position
// trap: for orbit trap coloring : https://en.wikipedia.org/wiki/Orbit_trap
// return: minimum distance to the mandelbulb surface
__device__ float md(vec3 p, float &trap)
{
    vec3 v = p;
    float dr = 1.f;           // |v'|
    float r = glm::length(v); // r = |v| = sqrt(x^2 + y^2 + z^2)
    trap = r;

    // Precompute power - 1 (used in every iteration)
    const float power_minus_1 = d_power - 1.f;
    const float half_log_bailout = 0.5f * logf(d_bailout); // Early exit optimization

    for (int i = 0; i < d_md_iter; ++i)
    {
        // Early exit if we're far enough
        if (r > d_bailout)
            break;

        float theta = glm::atan(v.y, v.x) * d_power;
        float phi = glm::asin(v.z / r) * d_power;

        // Use FMA: dr = d_power * pow(r, d_power-1) * dr + 1
        float r_pow = glm::pow(r, power_minus_1);
        dr = fmaf(d_power * r_pow, dr, 1.f);

        // Calculate powered radius
        float r_power = r * r_pow; // r^d_power = r * r^(d_power-1)
        float cos_theta = cosf(theta);
        float sin_theta = sinf(theta);
        float cos_phi = cosf(phi);
        float sin_phi = sinf(phi);

        // Use FMA for vector update: v = p + r_power * vec3(...)
        v.x = fmaf(r_power, cos_theta * cos_phi, p.x);
        v.y = fmaf(r_power, cos_phi * sin_theta, p.y);
        v.z = fmaf(r_power, -sin_phi, p.z);

        // orbit trap for coloring
        trap = glm::min(trap, r);

        r = glm::length(v); // update r
    }
    return 0.5f * logf(r) * r / dr; // mandelbulb's DE function
}

// scene mapping
__device__ float map(vec3 p, float &trap, int &ID)
{
    vec2 rt = vec2(cos(pi / 2.f), sin(pi / 2.f));
    vec3 rp = mat3(1.f, 0.f, 0.f, 0.f, rt.x, -rt.y, 0.f, rt.y, rt.x) *
              p; // rotation matrix, rotate 90 deg (pi/2) along the X-axis
    ID = 1;
    return md(rp, trap);
}

// dummy function
// becase we dont need to know the ordit trap or the object ID when we are calculating the surface
// normal
__device__ float map(vec3 p)
{
    float dmy; // dummy
    int dmy2;  // dummy2
    return map(p, dmy, dmy2);
}

// simple palette function (borrowed from Inigo Quilez)
// see: https://www.shadertoy.com/view/ll2GD3
__device__ vec3 pal(float t, vec3 a, vec3 b, vec3 c, vec3 d)
{
    return a + b * glm::cos(2.f * pi * (c * t + d));
}

// second march: cast shadow
// also borrowed from Inigo Quilez
// see: http://www.iquilezles.org/www/articles/rmshadows/rmshadows.htm
__device__ float softshadow(vec3 ro, vec3 rd, float k)
{
    float res = 1.0f;
    float t = 0.f; // total distance
    for (int i = 0; i < d_shadow_step; ++i)
    {
        float h = map(ro + rd * t);
        res = glm::min(
            res, k * h / t); // closer to the objects, k*h/t terms will produce darker shadow
        if (res < 0.02f)
            return 0.02f;
        t += glm::clamp(h, .001f, d_step_limiter); // move ray
    }
    return glm::clamp(res, .02f, 1.f);
}

// use gradient to calc surface normal
__device__ vec3 calcNor(vec3 p)
{
    vec2 e = vec2(d_eps, 0.f);
    vec3 ex = vec3(e.x, e.y, e.y);                        // (eps, 0, 0)
    vec3 ey = vec3(e.y, e.x, e.y);                        // (0, eps, 0)
    vec3 ez = vec3(e.y, e.y, e.x);                        // (0, 0, eps)
    return glm::normalize(vec3(map(p + ex) - map(p - ex), // dx
                               map(p + ey) - map(p - ey), // dy
                               map(p + ez) - map(p - ez)  // dz
                               ));
}

// first march: find object's surface
__device__ float trace(vec3 ro, vec3 rd, float &trap, int &ID)
{
    float t = 0.f;   // total distance
    float len = 0.f; // current distance

    for (int i = 0; i < d_ray_step; ++i)
    {
        len = map(ro + rd * t, trap,
                  ID); // get minimum distance from current ray position to the object's surface
        if (glm::abs(len) < d_eps || t > d_far_plane)
            break;
        t += len * d_ray_multiplier;
    }
    return t < d_far_plane
               ? t
               : -1.f; // if exceeds the far plane then return -1 which means the ray missed a shot
}

// Render kernel - ray marching for each pixel
__launch_bounds__(128, 6) // Configured for 128 threads per block
    __global__ void renderKernel(unsigned char *raw_image, /*unsigned char **image ,*/ int width, int height)
{
    // Calculate 2D coordinates
    int j = blockIdx.x * blockDim.x + threadIdx.x; // x (column)
    int i = blockIdx.y * blockDim.y + threadIdx.y; // y (row)

    // Shared memory for camera vectors (computed once per block)
    __shared__ vec3 cf;
    __shared__ vec3 cs;
    __shared__ vec3 cu;
    __shared__ vec3 ro;
    __shared__ vec3 sd;      // sun direction (lighting)
    __shared__ vec3 sc;      // light color
    __shared__ float inv_AA; // 1.0 / d_AA

    // First thread in block computes camera vectors
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        ro = d_camera_pos;                                        // ray (camera) origin
        vec3 ta = d_target_pos;                                   // target position
        cf = glm::normalize(ta - ro);                             // forward vector
        cs = glm::normalize(glm::cross(cf, vec3(0.f, 1.f, 0.f))); // right (side) vector
        cu = glm::normalize(glm::cross(cs, cf));                  // up vector

        // Pre-compute lighting constants
        sd = glm::normalize(d_camera_pos); // sun direction
        sc = vec3(1.f, .9f, .717f);        // light color
        inv_AA = 1.0f / (float)d_AA;       // inverse AA for optimization
    }
    __syncthreads();

    // Boundary check
    if (i >= height || j >= width)
        return;

    // Declare color accumulator variables
    float fcol_r = 0.0f;
    float fcol_g = 0.0f;
    float fcol_b = 0.0f;

// Anti-aliasing loop - manually unrolled (3x3 = 9 iterations)
#pragma unroll
    for (int m = 0; m < d_AA; m++)
    {
#pragma unroll
        for (int n = 0; n < d_AA; n++)
        {
            vec2 p = vec2(j, i) + vec2(m, n) * inv_AA; // Use pre-computed inverse

            //---convert screen space coordinate to (-ap~ap, -1~1)
            // ap = aspect ratio = width/height
            vec2 uv = (-d_iResolution + 2.f * p) / d_iResolution.y;
            uv.y *= -1.f; // flip upside down
            //---

            // Use shared memory camera vectors
            vec3 rd = glm::normalize(uv.x * cs + uv.y * cu + d_FOV * cf); // ray direction
            //---

            //---marching
            float trap; // orbit trap
            int objID;  // the object id intersected with
            float d = trace(ro, rd, trap, objID);
            //---

            //---coloring
            if (d < 0.f)
            { // miss (hit sky)
              // Sky color (black) - directly add to accumulator
              // No need to create vec3 col
            }
            else
            {
                vec3 pos = ro + rd * d;             // hit position
                vec3 nr = calcNor(pos);             // get surface normal
                vec3 hal = glm::normalize(sd - rd); // blinn-phong lighting model (vector h)

                // Pre-compute dot products
                float sd_nr = glm::dot(sd, nr);
                float nr_hal = glm::dot(nr, hal);

                // use orbit trap to get the color
                vec3 col = pal(trap - .4f, vec3(.5f), vec3(.5f), vec3(1.f),
                               vec3(.0f, .1f, .2f)); // diffuse color

                // simple blinn phong lighting model
                float amb =
                    (0.7f + 0.3f * nr.y) *
                    (0.2f + 0.8f * glm::clamp(0.05f * logf(trap), 0.0f, 1.0f)); // self occlution
                float sdw = softshadow(pos + .001f * nr, sd, 16.f);             // shadow
                float dif = glm::clamp(sd_nr, 0.f, 1.f) * sdw;                  // diffuse
                float spe = powf(glm::clamp(nr_hal, 0.f, 1.f), 32.f) * dif;     // self shadow

                // Apply lighting (reduce vec3 usage)
                vec3 ambc = vec3(0.3f);                              // ambient color
                col *= ambc * (.05f + .95f * amb) + sc * dif * 0.8f; // combined lighting

                col = glm::pow(col, vec3(.7f, .9f, 1.f)); // fake SSS (subsurface scattering)
                col += spe * 0.8f;                        // specular

                // gamma correction and accumulate
                col = glm::clamp(glm::pow(col, vec3(.4545f)), 0.f, 1.f);
                fcol_r += col.r;
                fcol_g += col.g;
                fcol_b += col.b;
            }
            //---
        }
    }

    // Average the accumulated colors
    fcol_r /= (float)(d_AA * d_AA);
    fcol_g /= (float)(d_AA * d_AA);
    fcol_b /= (float)(d_AA * d_AA);

    // Convert float (0~1) to unsigned char (0~255)
    fcol_r *= 255.0f;
    fcol_g *= 255.0f;
    fcol_b *= 255.0f;

    // Write to image - Coalesced memory write (32-bit aligned)
    int pixel_idx = i * width + j;
    uchar4 *image_ptr = (uchar4 *)raw_image;
    image_ptr[pixel_idx] = make_uchar4(
        (unsigned char)fcol_r,
        (unsigned char)fcol_g,
        (unsigned char)fcol_b,
        255);
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
    vec3 camera_pos = vec3(atof(argv[1]), atof(argv[2]), atof(argv[3]));
    vec3 target_pos = vec3(atof(argv[4]), atof(argv[5]), atof(argv[6]));
    width = atoi(argv[7]);
    height = atoi(argv[8]);
    const char *filename = argv[9];

    printf("Camera: (%.3f, %.3f, %.3f)\n", camera_pos.x, camera_pos.y, camera_pos.z);
    printf("Target: (%.3f, %.3f, %.3f)\n", target_pos.x, target_pos.y, target_pos.z);
    printf("Image size: %d x %d\n", width, height);

    // Step 2.5: Initialize constant memory on GPU
    int AA = 3;
    float power = 8.0f;
    float md_iter = 24.0f;
    float ray_step = 10000.0f;
    float shadow_step = 1500.0f;
    float step_limiter = 0.2f;
    float ray_multiplier = 0.1f;
    float bailout = 2.0f;
    float eps = 0.0005f;
    float FOV = 1.5f;
    float far_plane = 100.0f;
    vec2 iResolution = vec2(width, height);

    cudaMemcpyToSymbol(d_AA, &AA, sizeof(int));
    cudaMemcpyToSymbol(d_power, &power, sizeof(float));
    cudaMemcpyToSymbol(d_md_iter, &md_iter, sizeof(float));
    cudaMemcpyToSymbol(d_ray_step, &ray_step, sizeof(float));
    cudaMemcpyToSymbol(d_shadow_step, &shadow_step, sizeof(float));
    cudaMemcpyToSymbol(d_step_limiter, &step_limiter, sizeof(float));
    cudaMemcpyToSymbol(d_ray_multiplier, &ray_multiplier, sizeof(float));
    cudaMemcpyToSymbol(d_bailout, &bailout, sizeof(float));
    cudaMemcpyToSymbol(d_eps, &eps, sizeof(float));
    cudaMemcpyToSymbol(d_FOV, &FOV, sizeof(float));
    cudaMemcpyToSymbol(d_far_plane, &far_plane, sizeof(float));
    cudaMemcpyToSymbol(d_camera_pos, &camera_pos, sizeof(vec3));
    cudaMemcpyToSymbol(d_target_pos, &target_pos, sizeof(vec3));
    cudaMemcpyToSymbol(d_iResolution, &iResolution, sizeof(vec2));

    // Step 3: Allocate host memory
    size_t image_size = width * height * 4 * sizeof(unsigned char); // RGBA
    h_raw_image = (unsigned char *)malloc(image_size);
    // h_image_ptr = (unsigned char **)malloc(height * sizeof(unsigned char *));

    // for (int i = 0; i < height; i++)
    // {
    //     h_image_ptr[i] = h_raw_image + i * width * 4;
    // }

    // Step 4: Allocate device memory
    unsigned char *d_raw_image;
    // unsigned char **d_image_ptr;
    cudaMalloc(&d_raw_image, image_size);
    // cudaMalloc(&d_image_ptr, height * sizeof(unsigned char *));

    // Copy row pointers to device (this is tricky, need to update pointers)
    // unsigned char **h_temp_ptr = (unsigned char **)malloc(height * sizeof(unsigned char *));
    // for (int i = 0; i < height; i++)
    // {
    //     h_temp_ptr[i] = d_raw_image + i * width * 4; // Device pointers
    // }
    // cudaMemcpy(d_image_ptr, h_temp_ptr, height * sizeof(unsigned char *), cudaMemcpyHostToDevice);
    // free(h_temp_ptr);

    // Step 5: Configure 2D grid and block dimensions
    dim3 blockSize(8, 16); // 256 threads per block
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y);

    // printf("Launching kernel: grid(%d, %d), block(%d, %d)\n",
    //        gridSize.x, gridSize.y, blockSize.x, blockSize.y);

    // Step 6: Launch kernel
    renderKernel<<<gridSize, blockSize>>>(d_raw_image, /*d_image_ptr,*/ width, height);

    // Synchronize and check for errors
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Step 7: Copy results back to CPU
    cudaMemcpy(h_raw_image, d_raw_image, image_size, cudaMemcpyDeviceToHost);

    // Step 8: Save PNG file
    unsigned error = lodepng_encode32_file(filename, h_raw_image, width, height);
    if (error)
    {
        printf("PNG error %u: %s\n", error, lodepng_error_text(error));
        return 1;
    }

    printf("Image saved to %s\n", filename);

    // Step 9: Free memory
    free(h_raw_image);
    // free(h_image_ptr);
    cudaFree(d_raw_image);
    // cudaFree(d_image_ptr);

    // Clean up events
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);

    return 0;
}