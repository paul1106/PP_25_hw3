// hw3.cu - Stage 1: Basic Framework
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// GLM vector math library
#define GLM_COMPILER 0
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

#define pi 3.1415926535897932384626433832795

typedef glm::dvec2 vec2; // doube precision 2D vector (x, y) or (u, v)
typedef glm::dvec3 vec3; // 3D vector (x, y, z) or (r, g, b)
typedef glm::dvec4 vec4; // 4D vector (x, y, z, w)
typedef glm::dmat3 mat3; // 3x3 matrix

// Host variables
unsigned int width;       // image width
unsigned int height;      // image height
unsigned char *h_raw_image;  // 1D image
unsigned char **h_image_ptr; // 2D image

// Device constant memory (GPU can access, read-only, fast)
__constant__ int d_AA;
__constant__ double d_power;
__constant__ double d_md_iter;
__constant__ double d_ray_step;
__constant__ double d_shadow_step;
__constant__ double d_step_limiter;
__constant__ double d_ray_multiplier;
__constant__ double d_bailout;
__constant__ double d_eps;
__constant__ double d_FOV;
__constant__ double d_far_plane;
__constant__ vec3 d_camera_pos;
__constant__ vec3 d_target_pos;
__constant__ vec2 d_iResolution;

// mandelbulb distance function (DE)
// v = v^8 + c
// p: current position
// trap: for orbit trap coloring : https://en.wikipedia.org/wiki/Orbit_trap
// return: minimum distance to the mandelbulb surface
__device__ double md(vec3 p, double &trap)
{
    vec3 v = p;
    double dr = 1.;            // |v'|
    double r = glm::length(v); // r = |v| = sqrt(x^2 + y^2 + z^2)
    trap = r;

    for (int i = 0; i < d_md_iter; ++i)
    {
        double theta = glm::atan(v.y, v.x) * d_power;
        double phi = glm::asin(v.z / r) * d_power;
        dr = d_power * glm::pow(r, d_power - 1.) * dr + 1.;
        v = p + glm::pow(r, d_power) *
                    vec3(cos(theta) * cos(phi), cos(phi) * sin(theta), -sin(phi)); // update vk+1

        // orbit trap for coloring
        trap = glm::min(trap, r);

        r = glm::length(v); // update r
        if (r > d_bailout)
            break; // if escaped
    }
    return 0.5 * log(r) * r / dr; // mandelbulb's DE function
}

// scene mapping
__device__ double map(vec3 p, double &trap, int &ID)
{
    vec2 rt = vec2(cos(pi / 2.), sin(pi / 2.));
    vec3 rp = mat3(1., 0., 0., 0., rt.x, -rt.y, 0., rt.y, rt.x) *
              p; // rotation matrix, rotate 90 deg (pi/2) along the X-axis
    ID = 1;
    return md(rp, trap);
}

// dummy function
// becase we dont need to know the ordit trap or the object ID when we are calculating the surface
// normal
__device__ double map(vec3 p)
{
    double dmy; // dummy
    int dmy2;   // dummy2
    return map(p, dmy, dmy2);
}

// simple palette function (borrowed from Inigo Quilez)
// see: https://www.shadertoy.com/view/ll2GD3
__device__ vec3 pal(double t, vec3 a, vec3 b, vec3 c, vec3 d)
{
    return a + b * glm::cos(2. * pi * (c * t + d));
}

// second march: cast shadow
// also borrowed from Inigo Quilez
// see: http://www.iquilezles.org/www/articles/rmshadows/rmshadows.htm
__device__ double softshadow(vec3 ro, vec3 rd, double k)
{
    double res = 1.0;
    double t = 0.; // total distance
    for (int i = 0; i < d_shadow_step; ++i)
    {
        double h = map(ro + rd * t);
        res = glm::min(
            res, k * h / t); // closer to the objects, k*h/t terms will produce darker shadow
        if (res < 0.02)
            return 0.02;
        t += glm::clamp(h, .001, d_step_limiter); // move ray
    }
    return glm::clamp(res, .02, 1.);
}

// use gradient to calc surface normal
__device__ vec3 calcNor(vec3 p)
{
    vec2 e = vec2(d_eps, 0.);
    vec3 ex = vec3(e.x, e.y, e.y);  // (eps, 0, 0)
    vec3 ey = vec3(e.y, e.x, e.y);  // (0, eps, 0)
    vec3 ez = vec3(e.y, e.y, e.x);  // (0, 0, eps)
    return glm::normalize(vec3(map(p + ex) - map(p - ex), // dx
                               map(p + ey) - map(p - ey), // dy
                               map(p + ez) - map(p - ez)  // dz
                               ));
}

// first march: find object's surface
__device__ double trace(vec3 ro, vec3 rd, double &trap, int &ID)
{
    double t = 0;   // total distance
    double len = 0; // current distance

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
               : -1.; // if exceeds the far plane then return -1 which means the ray missed a shot
}

// Render kernel - ray marching for each pixel
__global__ void renderKernel(unsigned char *raw_image, unsigned char **image, int width, int height)
{
    // Calculate 2D coordinates
    int j = blockIdx.x * blockDim.x + threadIdx.x; // x (column)
    int i = blockIdx.y * blockDim.y + threadIdx.y; // y (row)

    // Boundary check
    if (i >= height || j >= width)
        return;

    // Declare color accumulator variables
    double fcol_r = 0.0;
    double fcol_g = 0.0;
    double fcol_b = 0.0;

    // Anti-aliasing loop
    for (int m = 0; m < d_AA; m++)
    {
        for (int n = 0; n < d_AA; n++)
        {
            vec2 p = vec2(j, i) + vec2(m, n) / (double)d_AA;

            //---convert screen space coordinate to (-ap~ap, -1~1)
            // ap = aspect ratio = width/height
            vec2 uv = (-d_iResolution + 2. * p) / d_iResolution.y;
            uv.y *= -1; // flip upside down
            //---

            //---create camera
            vec3 ro = d_camera_pos;              // ray (camera) origin
            vec3 ta = d_target_pos;              // target position
            vec3 cf = glm::normalize(ta - ro); // forward vector
            vec3 cs =
                glm::normalize(glm::cross(cf, vec3(0., 1., 0.)));       // right (side) vector
            vec3 cu = glm::normalize(glm::cross(cs, cf));               // up vector
            vec3 rd = glm::normalize(uv.x * cs + uv.y * cu + d_FOV * cf); // ray direction
            //---

            //---marching
            double trap; // orbit trap
            int objID;   // the object id intersected with
            double d = trace(ro, rd, trap, objID);
            //---

            //---lighting
            vec3 col(0.);                         // color
            vec3 sd = glm::normalize(d_camera_pos); // sun direction (directional light)
            vec3 sc = vec3(1., .9, .717);         // light color
            //---

            //---coloring
            if (d < 0.)
            {                   // miss (hit sky)
                col = vec3(0.); // sky color (black)
            }
            else
            {
                vec3 pos = ro + rd * d;             // hit position
                vec3 nr = calcNor(pos);             // get surface normal
                vec3 hal = glm::normalize(sd - rd); // blinn-phong lighting model (vector
                                                    // h)
                // for more info:
                // https://en.wikipedia.org/wiki/Blinn%E2%80%93Phong_shading_model

                // use orbit trap to get the color
                col = pal(trap - .4, vec3(.5), vec3(.5), vec3(1.),
                          vec3(.0, .1, .2)); // diffuse color
                vec3 ambc = vec3(0.3);       // ambient color
                double gloss = 32.;          // specular gloss

                // simple blinn phong lighting model
                double amb =
                    (0.7 + 0.3 * nr.y) *
                    (0.2 + 0.8 * glm::clamp(0.05 * log(trap), 0.0, 1.0)); // self occlution
                double sdw = softshadow(pos + .001 * nr, sd, 16.);        // shadow
                double dif = glm::clamp(glm::dot(sd, nr), 0., 1.) * sdw;  // diffuse
                double spe = glm::pow(glm::clamp(glm::dot(nr, hal), 0., 1.), gloss) *
                             dif; // self shadow

                vec3 lin(0.);
                lin += ambc * (.05 + .95 * amb); // ambient color * ambient
                lin += sc * dif * 0.8;           // diffuse * light color * light intensity
                col *= lin;

                col = glm::pow(col, vec3(.7, .9, 1.)); // fake SSS (subsurface scattering)
                col += spe * 0.8;                      // specular
            }
            //---

            col = glm::clamp(glm::pow(col, vec3(.4545)), 0., 1.); // gamma correction
            // fcol += vec4(col, 1.);
            fcol_r += col.r;
            fcol_g += col.g;
            fcol_b += col.b;
        }
    }

    // Average the accumulated colors
    fcol_r /= (double)(d_AA * d_AA);
    fcol_g /= (double)(d_AA * d_AA);
    fcol_b /= (double)(d_AA * d_AA);

    // Convert double (0~1) to unsigned char (0~255)
    fcol_r *= 255.0;
    fcol_g *= 255.0;
    fcol_b *= 255.0;

    // Write to image
    image[i][4 * j + 0] = (unsigned char)fcol_r; // r
    image[i][4 * j + 1] = (unsigned char)fcol_g; // g
    image[i][4 * j + 2] = (unsigned char)fcol_b; // b
    image[i][4 * j + 3] = 255;                   // a
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
    double power = 8.0;
    double md_iter = 24;
    double ray_step = 10000;
    double shadow_step = 1500;
    double step_limiter = 0.2;
    double ray_multiplier = 0.1;
    double bailout = 2.0;
    double eps = 0.0005;
    double FOV = 1.5;
    double far_plane = 100.0;
    vec2 iResolution = vec2(width, height);

    cudaMemcpyToSymbol(d_AA, &AA, sizeof(int));
    cudaMemcpyToSymbol(d_power, &power, sizeof(double));
    cudaMemcpyToSymbol(d_md_iter, &md_iter, sizeof(double));
    cudaMemcpyToSymbol(d_ray_step, &ray_step, sizeof(double));
    cudaMemcpyToSymbol(d_shadow_step, &shadow_step, sizeof(double));
    cudaMemcpyToSymbol(d_step_limiter, &step_limiter, sizeof(double));
    cudaMemcpyToSymbol(d_ray_multiplier, &ray_multiplier, sizeof(double));
    cudaMemcpyToSymbol(d_bailout, &bailout, sizeof(double));
    cudaMemcpyToSymbol(d_eps, &eps, sizeof(double));
    cudaMemcpyToSymbol(d_FOV, &FOV, sizeof(double));
    cudaMemcpyToSymbol(d_far_plane, &far_plane, sizeof(double));
    cudaMemcpyToSymbol(d_camera_pos, &camera_pos, sizeof(vec3));
    cudaMemcpyToSymbol(d_target_pos, &target_pos, sizeof(vec3));
    cudaMemcpyToSymbol(d_iResolution, &iResolution, sizeof(vec2));

    // Step 3: Allocate host memory
    size_t image_size = width * height * 4 * sizeof(unsigned char); // RGBA
    h_raw_image = (unsigned char *)malloc(image_size);
    h_image_ptr = (unsigned char **)malloc(height * sizeof(unsigned char *));

    for (int i = 0; i < height; i++)
    {
        h_image_ptr[i] = h_raw_image + i * width * 4;
    }

    // Step 4: Allocate device memory
    unsigned char *d_raw_image;
    unsigned char **d_image_ptr;
    cudaMalloc(&d_raw_image, image_size);
    cudaMalloc(&d_image_ptr, height * sizeof(unsigned char *));

    // Copy row pointers to device (this is tricky, need to update pointers)
    unsigned char **h_temp_ptr = (unsigned char **)malloc(height * sizeof(unsigned char *));
    for (int i = 0; i < height; i++)
    {
        h_temp_ptr[i] = d_raw_image + i * width * 4;  // Device pointers
    }
    cudaMemcpy(d_image_ptr, h_temp_ptr, height * sizeof(unsigned char *), cudaMemcpyHostToDevice);
    free(h_temp_ptr);

    // Step 5: Configure 2D grid and block dimensions
    dim3 blockSize(16, 16); // 256 threads per block
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y);

    printf("Launching kernel: grid(%d, %d), block(%d, %d)\n",
           gridSize.x, gridSize.y, blockSize.x, blockSize.y);

    // Step 6: Launch kernel
    renderKernel<<<gridSize, blockSize>>>(d_raw_image, d_image_ptr, width, height);

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
    free(h_image_ptr);
    cudaFree(d_raw_image);
    cudaFree(d_image_ptr);

    return 0;
}