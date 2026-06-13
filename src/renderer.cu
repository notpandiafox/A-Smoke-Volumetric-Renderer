#include "common.cuh"
#include "renderer.h"
#include <cstdio>

#ifdef USE_NANOVDB
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/SampleFromVoxels.h>
#endif

namespace {
cudaArray_t        g_denArray = nullptr;
cudaTextureObject_t g_denTex  = 0;
int3               g_res;
}

__device__ float sampleDensity(cudaTextureObject_t tex, float3 p, float3 bmin, float3 bmax)
{
    float3 uvw = (p - bmin) / (bmax - bmin);
    return tex3D<float>(tex, uvw.x, uvw.y, uvw.z);
}

#ifdef USE_NANOVDB
__device__ float sampleDensityVDB(const nanovdb::NanoGrid<float>* grid, float3 p)
{
    auto acc = grid->getAccessor();
    auto sampler = nanovdb::createSampler<1>(acc);  // 1 = trilinear
    nanovdb::Vec3f idx = grid->worldToIndexF(nanovdb::Vec3f(p.x, p.y, p.z));
    return sampler(idx);
}
#endif

template <bool kUseVDB>
__device__ void integrate(const DRay& ray, float tMin, float tMax, float3& L, float& T, cudaTextureObject_t denTex,
     const void* vdbGrid, const RenderParams rp)
{
    const float stride = rp.stepSize;
    const float sigma_t = rp.sigma_a + rp.sigma_s;
    int numSteps = (int)ceilf((tMax - tMin) / stride);
    numSteps = min(numSteps, rp.maxSteps);

    float3 Lvol = make_float3(0, 0, 0);
    float  Tvol = 1.0f;
    const float ph = phaseHG(-ray.dir, rp.lightDir, rp.hgG);

    for (int n = 0; n < numSteps; ++n) {
        float t = tMin + stride * (n + 0.5f);
        float3 sp = ray.at(t);

        float density;
#ifdef USE_NANOVDB
        if (kUseVDB)
            density = sampleDensityVDB((const nanovdb::NanoGrid<float>*)vdbGrid, sp);
        else
#endif
            density = sampleDensity(denTex, sp, rp.bmin, rp.bmax);

        if (density > 1e-4f) {
            Tvol *= expf(-stride * density * sigma_t);

            
            DRay lightRay(sp, rp.lightDir);
            float tlMin, tlMax;
            if (raybox(lightRay, rp.bmin, rp.bmax, tlMin, tlMax) && tlMax > 0.0f) {
                const float ls = stride * rp.lightStrideMult;
                int nl = min((int)ceilf(tlMax / ls), rp.maxLightSteps);
                float densityLight = 0.0f;
                for (int k = 0; k < nl; ++k) {
                    float3 lp = lightRay.at(ls * (k + 0.5f));
#ifdef USE_NANOVDB
                    if (kUseVDB)
                        densityLight += sampleDensityVDB((const nanovdb::NanoGrid<float>*)vdbGrid, lp);
                    else
#endif
                        densityLight += sampleDensity(denTex, lp, rp.bmin, rp.bmax);
                }
                float lightAtt = expf(-densityLight * ls * sigma_t);
                Lvol = Lvol + rp.lightColor * (lightAtt * ph * rp.sigma_s * Tvol * stride * density);
            }
        }

        
        if (Tvol < 0.005f) { Tvol = 0.0f; break; }
    }

    L = Lvol;
    T = Tvol;
}

template <bool kUseVDB>
__global__ void renderKernel(uchar4* out, const void* vdbGrid, cudaTextureObject_t denTex, const RenderParams rp)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= rp.width || j >= rp.height) return;

    
    float3 rd;
    rd.x = (2.0f * (i + 0.5f) / rp.width - 1.0f) * rp.focal;
    rd.y = (1.0f - 2.0f * (j + 0.5f) / rp.height) * rp.focal / rp.aspect;
    rd.z = -1.0f;
    
    float3 dir = make_float3(
        rp.c2w[0]*rd.x + rp.c2w[4]*rd.y + rp.c2w[ 8]*rd.z,
        rp.c2w[1]*rd.x + rp.c2w[5]*rd.y + rp.c2w[ 9]*rd.z,
        rp.c2w[2]*rd.x + rp.c2w[6]*rd.y + rp.c2w[10]*rd.z);
    dir = normalize(dir);
    float3 orig = make_float3(rp.c2w[12], rp.c2w[13], rp.c2w[14]);

    DRay ray(orig, dir);

    float3 L = make_float3(0, 0, 0);
    float  T = 1.0f;
    float tmin, tmax;
    if (raybox(ray, rp.bmin, rp.bmax, tmin, tmax) && tmax > 0.0f) {
        tmin = fmaxf(tmin, 0.0f);                      
        integrate<kUseVDB>(ray, tmin, tmax, L, T, denTex, vdbGrid, rp);
    }

    float3 c = rp.background * T + L;
    
    c.x = powf(c.x / (1.0f + c.x), 1.0f / 2.2f);
    c.y = powf(c.y / (1.0f + c.y), 1.0f / 2.2f);
    c.z = powf(c.z / (1.0f + c.z), 1.0f / 2.2f);

    out[j * rp.width + i] = make_uchar4(
        (unsigned char)(__saturatef(c.x) * 255.0f),
        (unsigned char)(__saturatef(c.y) * 255.0f),
        (unsigned char)(__saturatef(c.z) * 255.0f), 255);
}


void initDensityTexture(int3 res)
{
    g_res = res;
    cudaChannelFormatDesc cd = cudaCreateChannelDesc<float>();
    cudaExtent ext = make_cudaExtent(res.x, res.y, res.z);
    CUDA_CHECK(cudaMalloc3DArray(&g_denArray, &cd, ext));

    cudaResourceDesc rd = {};
    rd.resType = cudaResourceTypeArray;
    rd.res.array.array = g_denArray;

    cudaTextureDesc td = {};
    td.addressMode[0] = td.addressMode[1] = td.addressMode[2] = cudaAddressModeClamp;
    td.filterMode = cudaFilterModeLinear;      
    td.readMode   = cudaReadModeElementType;
    td.normalizedCoords = 1;

    CUDA_CHECK(cudaCreateTextureObject(&g_denTex, &rd, &td, nullptr));
}

void uploadDensity(const float* d_linearDensity)
{
    cudaMemcpy3DParms p = {};
    p.srcPtr = make_cudaPitchedPtr((void*)d_linearDensity,
                                   g_res.x * sizeof(float), g_res.x, g_res.y);
    p.dstArray = g_denArray;
    p.extent = make_cudaExtent(g_res.x, g_res.y, g_res.z);
    p.kind = cudaMemcpyDeviceToDevice;
    CUDA_CHECK(cudaMemcpy3D(&p));
}

void launchRender(uchar4* d_out, const void* d_vdbGrid, bool useVDB, const RenderParams& rp)
{
    dim3 block(16, 16);
    dim3 grid((rp.width + 15) / 16, (rp.height + 15) / 16);
    if (useVDB)
        renderKernel<true ><<<grid, block>>>(d_out, d_vdbGrid, g_denTex, rp);
    else
        renderKernel<false><<<grid, block>>>(d_out, d_vdbGrid, g_denTex, rp);
    CUDA_CHECK(cudaGetLastError());
}

void destroyRenderer()
{
    if (g_denTex)   cudaDestroyTextureObject(g_denTex);
    if (g_denArray) cudaFreeArray(g_denArray);
}
