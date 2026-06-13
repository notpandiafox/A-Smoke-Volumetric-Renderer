#pragma once
=#include <cuda_runtime.h>
#include <cmath>

#define PI_F 3.14159265358979f

__host__ __device__ inline float3 operator+(float3 a, float3 b) { return make_float3(a.x+b.x, a.y+b.y, a.z+b.z); }
__host__ __device__ inline float3 operator-(float3 a, float3 b) { return make_float3(a.x-b.x, a.y-b.y, a.z-b.z); }
__host__ __device__ inline float3 operator*(float3 a, float s)  { return make_float3(a.x*s, a.y*s, a.z*s); }
__host__ __device__ inline float3 operator*(float s, float3 a)  { return a * s; }
__host__ __device__ inline float3 operator*(float3 a, float3 b) { return make_float3(a.x*b.x, a.y*b.y, a.z*b.z); }
__host__ __device__ inline float3 operator/(float3 a, float3 b) { return make_float3(a.x/b.x, a.y/b.y, a.z/b.z); }
__host__ __device__ inline float3 operator/(float s, float3 a)  { return make_float3(s/a.x, s/a.y, s/a.z); }
__host__ __device__ inline float3 operator-(float3 a)           { return make_float3(-a.x, -a.y, -a.z); }
__host__ __device__ inline float  dot(float3 a, float3 b)       { return a.x*b.x + a.y*b.y + a.z*b.z; }
__host__ __device__ inline float3 normalize(float3 v) {
    float inv = rsqrtf(dot(v, v));
    return v * inv;
}

struct DRay {
    float3 orig, dir, invDir;
    int sign[3];
    __host__ __device__ DRay(float3 o, float3 d) : orig(o), dir(d) {
        invDir = 1.0f / d;
        sign[0] = (invDir.x < 0.0f);
        sign[1] = (invDir.y < 0.0f);
        sign[2] = (invDir.z < 0.0f);
    }
    __host__ __device__ float3 at(float t) const { return orig + dir * t; }
};

__host__ __device__ inline bool raybox(const DRay& ray, const float3 bmin, const float3 bmax, float& tmin, float& tmax)
{
    const float3 bounds[2] = { bmin, bmax };
    float a = bounds[    ray.sign[0]].x - ray.orig.x;
    float b = bounds[1 - ray.sign[0]].x - ray.orig.x;
    float c = bounds[    ray.sign[1]].y - ray.orig.y;
    float d = bounds[1 - ray.sign[1]].y - ray.orig.y;

    float x0 = (a == 0.0f) ? 0.0f : a * ray.invDir.x;
    float x1 = (b == 0.0f) ? 0.0f : b * ray.invDir.x;
    float y0 = (c == 0.0f) ? 0.0f : c * ray.invDir.y;
    float y1 = (d == 0.0f) ? 0.0f : d * ray.invDir.y;

    if (x0 > y1 || y0 > x1) return false;
    tmin = fmaxf(x0, y0);
    tmax = fminf(x1, y1);

    float e = bounds[    ray.sign[2]].z - ray.orig.z;
    float f = bounds[1 - ray.sign[2]].z - ray.orig.z;
    float z0 = (e == 0.0f) ? 0.0f : e * ray.invDir.z;
    float z1 = (f == 0.0f) ? 0.0f : f * ray.invDir.z;

    if (tmin > z1 || z0 > tmax) return false;
    tmin = fmaxf(z0, tmin);
    tmax = fminf(z1, tmax);
    return true;
}

// Henyey-Greenstein phase function
__device__ inline float phaseHG(float3 viewDir, float3 lightDir, float g)
{
    float costheta = dot(viewDir, lightDir);
    return 1.0f / (4.0f * PI_F) * (1.0f - g * g)
           / powf(1.0f + g * g - 2.0f * g * costheta, 1.5f);
}

#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), __FILE__, __LINE__); \
        exit(1); \
    } } while (0)
