#include "common.cuh"
#include "fluid_sim.h"
#include <cstdio>

static int3 g_res;
static float3* d_vel  = nullptr;   // velocity field
static float3* d_vel2 = nullptr;   // scratch
static float*  d_den  = nullptr;   // density 
static float*  d_den2 = nullptr;
static float*  d_prs  = nullptr;   // pressure
static float*  d_prs2 = nullptr;
static float*  d_div  = nullptr;   // divergence
static float3* d_curl = nullptr;   // vorticity

__device__ __forceinline__ int idx3(int x, int y, int z, int3 r) {
    x = min(max(x, 0), r.x - 1);
    y = min(max(y, 0), r.y - 1);
    z = min(max(z, 0), r.z - 1);
    return (z * r.y + y) * r.x + x;
}

template <typename T>
__device__ T sampleTrilinear(const T* f, float3 p, int3 r)
{
    p.x = fminf(fmaxf(p.x, 0.5f), r.x - 1.5f);
    p.y = fminf(fmaxf(p.y, 0.5f), r.y - 1.5f);
    p.z = fminf(fmaxf(p.z, 0.5f), r.z - 1.5f);
    int x0 = (int)p.x, y0 = (int)p.y, z0 = (int)p.z;
    float fx = p.x - x0, fy = p.y - y0, fz = p.z - z0;

    T c000 = f[idx3(x0,   y0,   z0,   r)], c100 = f[idx3(x0+1, y0,   z0,   r)];
    T c010 = f[idx3(x0,   y0+1, z0,   r)], c110 = f[idx3(x0+1, y0+1, z0,   r)];
    T c001 = f[idx3(x0,   y0,   z0+1, r)], c101 = f[idx3(x0+1, y0,   z0+1, r)];
    T c011 = f[idx3(x0,   y0+1, z0+1, r)], c111 = f[idx3(x0+1, y0+1, z0+1, r)];

    T c00 = c000*(1-fx) + c100*fx,  c10 = c010*(1-fx) + c110*fx;
    T c01 = c001*(1-fx) + c101*fx,  c11 = c011*(1-fx) + c111*fx;
    T c0  = c00*(1-fy)  + c10*fy,   c1  = c01*(1-fy)  + c11*fy;
    return c0*(1-fz) + c1*fz;
}

#define FLUID_KERNEL_HEAD(r) \
    int x = blockIdx.x * blockDim.x + threadIdx.x; \
    int y = blockIdx.y * blockDim.y + threadIdx.y; \
    int z = blockIdx.z * blockDim.z + threadIdx.z; \
    if (x >= r.x || y >= r.y || z >= r.z) return;  \
    int i = (z * r.y + y) * r.x + x;

__global__ void kAdvectVelocity(float3* dst, const float3* vel, float dt, int3 r)
{
    FLUID_KERNEL_HEAD(r);
    float3 p = make_float3(x + 0.5f, y + 0.5f, z + 0.5f) - vel[i] * dt;
    dst[i] = sampleTrilinear(vel, p, r);
}

__global__ void kAdvectDensity(float* dst, const float* den, const float3* vel,
                               float dt, float dissipation, int3 r)
{
    FLUID_KERNEL_HEAD(r);
    float3 p = make_float3(x + 0.5f, y + 0.5f, z + 0.5f) - vel[i] * dt;
    dst[i] = sampleTrilinear(den, p, r) * dissipation;
}


__global__ void kBuoyancy(float3* vel, const float* den, float dt,
                          float buoyancy, float weight, int3 r)
{
    FLUID_KERNEL_HEAD(r);
    float d = den[i];
    vel[i].y += dt * (buoyancy * d - weight * d);
}


__global__ void kCurl(float3* curl, const float3* v, int3 r)
{
    FLUID_KERNEL_HEAD(r);
    float3 dx = v[idx3(x+1,y,z,r)] - v[idx3(x-1,y,z,r)];
    float3 dy = v[idx3(x,y+1,z,r)] - v[idx3(x,y-1,z,r)];
    float3 dz = v[idx3(x,y,z+1,r)] - v[idx3(x,y,z-1,r)];
    curl[i] = make_float3(dy.z - dz.y, dz.x - dx.z, dx.y - dy.x) * 0.5f;
}


__global__ void kVorticityConfine(float3* vel, const float3* curl, float epsilon, float dt, int3 r)
{
    FLUID_KERNEL_HEAD(r);
    float3 c = curl[i];
    float mag = sqrtf(dot(c, c));
    
    float gx = sqrtf(dot(curl[idx3(x+1,y,z,r)], curl[idx3(x+1,y,z,r)]))
             - sqrtf(dot(curl[idx3(x-1,y,z,r)], curl[idx3(x-1,y,z,r)]));
    float gy = sqrtf(dot(curl[idx3(x,y+1,z,r)], curl[idx3(x,y+1,z,r)]))
             - sqrtf(dot(curl[idx3(x,y-1,z,r)], curl[idx3(x,y-1,z,r)]));
    float gz = sqrtf(dot(curl[idx3(x,y,z+1,r)], curl[idx3(x,y,z+1,r)]))
             - sqrtf(dot(curl[idx3(x,y,z-1,r)], curl[idx3(x,y,z-1,r)]));
    float3 N = make_float3(gx, gy, gz) * 0.5f;
    float len = sqrtf(dot(N, N)) + 1e-5f;
    N = N * (1.0f / len);
    // F = ε (N × ω)
    float3 F = make_float3(N.y*c.z - N.z*c.y, N.z*c.x - N.x*c.z, N.x*c.y - N.y*c.x);
    vel[i] = vel[i] + F * (epsilon * dt);
    (void)mag;
}


__global__ void kDivergence(float* divr, const float3* v, int3 r)
{
    FLUID_KERNEL_HEAD(r);
    divr[i] = 0.5f * ((v[idx3(x+1,y,z,r)].x - v[idx3(x-1,y,z,r)].x)
                    + (v[idx3(x,y+1,z,r)].y - v[idx3(x,y-1,z,r)].y)
                    + (v[idx3(x,y,z+1,r)].z - v[idx3(x,y,z-1,r)].z));
}


__global__ void kJacobiPressure(float* pOut, const float* pIn, const float* divr, int3 r)
{
    FLUID_KERNEL_HEAD(r);
    pOut[i] = (pIn[idx3(x-1,y,z,r)] + pIn[idx3(x+1,y,z,r)]
             + pIn[idx3(x,y-1,z,r)] + pIn[idx3(x,y+1,z,r)]
             + pIn[idx3(x,y,z-1,r)] + pIn[idx3(x,y,z+1,r)]
             - divr[i]) / 6.0f;
}


__global__ void kSubtractGradient(float3* v, const float* p, int3 r)
{
    FLUID_KERNEL_HEAD(r);
    v[i].x -= 0.5f * (p[idx3(x+1,y,z,r)] - p[idx3(x-1,y,z,r)]);
    v[i].y -= 0.5f * (p[idx3(x,y+1,z,r)] - p[idx3(x,y-1,z,r)]);
    v[i].z -= 0.5f * (p[idx3(x,y,z+1,r)] - p[idx3(x,y,z-1,r)]);
}


__global__ void kInjectSource(float* den, float3* vel, float t, int3 r)
{
    FLUID_KERNEL_HEAD(r);
    float cx = r.x * 0.5f + sinf(t * 0.7f) * r.x * 0.08f;
    float cz = r.z * 0.5f + cosf(t * 0.5f) * r.z * 0.08f;
    float dx = x - cx, dz = z - cz;
    float rad = r.x * 0.10f;
    if (y < r.y * 0.08f && dx*dx + dz*dz < rad*rad) {
        den[i] = fminf(den[i] + 0.6f, 2.0f);
        vel[i].y += 0.8f;
        vel[i].x += sinf(t * 1.3f) * 0.2f;
    }
}


void initFluidSim(int3 res)
{
    g_res = res;
    size_t n = (size_t)res.x * res.y * res.z;
    CUDA_CHECK(cudaMalloc(&d_vel,  n * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_vel2, n * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_den,  n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_den2, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_prs,  n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_prs2, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_div,  n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_curl, n * sizeof(float3)));
    CUDA_CHECK(cudaMemset(d_vel, 0, n * sizeof(float3)));
    CUDA_CHECK(cudaMemset(d_den, 0, n * sizeof(float)));
}



void simulateNavierStokes(float dt, float simTime, const FluidParams& fp)
{
    dim3 b(8, 8, 8);
    dim3 g((g_res.x + 7) / 8, (g_res.y + 7) / 8, (g_res.z + 7) / 8);

    kInjectSource<<<g, b>>>(d_den, d_vel, simTime, g_res);

    
    kAdvectVelocity<<<g, b>>>(d_vel2, d_vel, dt * fp.velScale, g_res);
    std::swap(d_vel, d_vel2);

    kBuoyancy<<<g, b>>>(d_vel, d_den, dt, fp.buoyancy, fp.weight, g_res);

    kCurl<<<g, b>>>(d_curl, d_vel, g_res);
    kVorticityConfine<<<g, b>>>(d_vel, d_curl, fp.vorticity, dt, g_res);

    
    kDivergence<<<g, b>>>(d_div, d_vel, g_res);
    CUDA_CHECK(cudaMemset(d_prs, 0, (size_t)g_res.x * g_res.y * g_res.z * sizeof(float)));
    for (int it = 0; it < fp.jacobiIters; ++it) {
        kJacobiPressure<<<g, b>>>(d_prs2, d_prs, d_div, g_res);
        std::swap(d_prs, d_prs2);
    }
    kSubtractGradient<<<g, b>>>(d_vel, d_prs, g_res);

    
    kAdvectDensity<<<g, b>>>(d_den2, d_den, d_vel, dt * fp.velScale, fp.dissipation, g_res);
    std::swap(d_den, d_den2);
}

const float* fluidDensityField() { return d_den; }

void destroyFluidSim()
{
    cudaFree(d_vel);  cudaFree(d_vel2);
    cudaFree(d_den);  cudaFree(d_den2);
    cudaFree(d_prs);  cudaFree(d_prs2);
    cudaFree(d_div);  cudaFree(d_curl);
}
