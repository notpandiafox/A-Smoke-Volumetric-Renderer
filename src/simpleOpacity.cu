#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <fstream>

#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <cmath>

// Image dimensions
#define WIDTH  640
#define HEIGHT 480

// Vec3 struct that works on both CPU and GPU
struct Vec3 {
    float x, y, z;

    __host__ __device__ Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3(float x, float y, float z) 
        : x(x), y(y), z(z) {}

    __host__ __device__ Vec3 operator+(const Vec3& v) const 
        { return Vec3(x+v.x, y+v.y, z+v.z); }
    
    __host__ __device__ Vec3 operator-(const Vec3& v) const 
        { return Vec3(x-v.x, y-v.y, z-v.z); }
    
    __host__ __device__ Vec3 operator*(float t) const 
        { return Vec3(x*t, y*t, z*t); }
    
    __host__ __device__ Vec3 operator*(const Vec3& v) const 
        { return Vec3(x*v.x, y*v.y, z*v.z); }

    __host__ __device__ float length() const 
        { return sqrtf(x*x + y*y + z*z); }

    __host__ __device__ Vec3 normalize() const {
        float len = length();
        return Vec3(x/len, y/len, z/len);
    }
};

// Sphere data — passed to GPU as a simple struct
struct Sphere {
    Vec3  center    { 0, 0, -4 };
    float radius    { 1.0f };
    float sigma_a   { 0.3f };
    Vec3  scatter   { 0.8f, 0.1f, 0.5f };
};

// Ray-sphere intersection — runs on GPU
__device__ bool intersectSphere( Vec3 ray_origin, Vec3 ray_dir, const Sphere& sphere, float& t0, float& t1)
{
    Vec3  oc = ray_origin - sphere.center;
    float a  = oc.x*ray_dir.x + oc.y*ray_dir.y + oc.z*ray_dir.z;
    float b  = oc.x*oc.x + oc.y*oc.y + oc.z*oc.z 
               - sphere.radius * sphere.radius;
    float discriminant = a*a - b;

    if (discriminant < 0.0f) return false;

    float sqrtDisc = sqrtf(discriminant);
    t0 = -a - sqrtDisc;
    t1 = -a + sqrtDisc;

    return true;
}

// Core render kernel — one thread per pixel
__global__ void renderKernel(Vec3* framebuffer, Sphere sphere)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= WIDTH || j >= HEIGHT) return;

    // Compute ray direction for this pixel
    float aspect = (float)WIDTH / (float)HEIGHT;
    float px = (2.0f * ((i + 0.5f) / WIDTH)  - 1.0f) * aspect;
    float py = (1.0f - 2.0f * ((j + 0.5f) / HEIGHT));
    Vec3 ray_origin(0, 0, 0);
    Vec3 ray_dir = Vec3(px, py, -1.0f).normalize();

    // Background color
    Vec3 background(0.572f, 0.772f, 0.921f);

    // Check intersection
    float t0, t1;
    Vec3 pixel_color;

    if (intersectSphere(ray_origin, ray_dir, sphere, t0, t1)) {
        // Compute entry and exit points
        Vec3 p1 = ray_origin + ray_dir * t0;
        Vec3 p2 = ray_origin + ray_dir * t1;

        // Distance ray travels through sphere
        float distance = (p2 - p1).length();

        // Beer's Law — how much light is transmitted
        float transmission = expf(-distance * sphere.sigma_a);

        // Blend background and scatter color
        pixel_color = background * transmission 
                    + sphere.scatter * (1.0f - transmission);
    }
    else {
        pixel_color = background;
    }

    // Write to framebuffer
    int idx = j * WIDTH + i;
    framebuffer[idx] = pixel_color;
}

// Save framebuffer as PPM image
void savePPM(const char* filename, Vec3* framebuffer)
{
    std::ofstream file(filename);
    file << "P3\n" << WIDTH << " " << HEIGHT << "\n255\n";
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        int r = (int)(fminf(framebuffer[i].x, 1.0f) * 255);
        int g = (int)(fminf(framebuffer[i].y, 1.0f) * 255);
        int b = (int)(fminf(framebuffer[i].z, 1.0f) * 255);
        file << r << " " << g << " " << b << "\n";
    }
}

int main()
{
    // Allocate framebuffer on GPU
    Vec3* d_framebuffer;
    cudaMalloc(&d_framebuffer, WIDTH * HEIGHT * sizeof(Vec3));

    // Set up sphere
    Sphere sphere;

    // Launch kernel — 16x16 threads per block
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (WIDTH  + blockSize.x - 1) / blockSize.x,
        (HEIGHT + blockSize.y - 1) / blockSize.y
    );

    renderKernel<<<gridSize, blockSize>>>(d_framebuffer, sphere);
    cudaDeviceSynchronize();

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " 
                  << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // Copy framebuffer back to CPU
    Vec3* h_framebuffer = new Vec3[WIDTH * HEIGHT];
    cudaMemcpy(h_framebuffer, d_framebuffer, 
               WIDTH * HEIGHT * sizeof(Vec3), 
               cudaMemcpyDeviceToHost);

    // Save image
    savePPM("output.ppm", h_framebuffer);
    std::cout << "Saved output.ppm" << std::endl;

    // Cleanup
    cudaFree(d_framebuffer);
    delete[] h_framebuffer;

    return 0;
}