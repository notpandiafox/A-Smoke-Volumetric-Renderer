#pragma once
#include <cuda_runtime.h>

struct RenderParams {
    int    width, height;
    float  focal, aspect;
    float  c2w[16];                 // camera-to-world matrix
    float3 bmin, bmax;              // volume bounds in world space
    float3 lightDir;
    float3 lightColor;
    float3 background;
    float  sigma_a = 0.5f;
    float  sigma_s = 0.5f;
    float  hgG     = 0.2f;          // slight forward scattering looks cloudier than 0
    float  stepSize = 0.35f;
    int    maxSteps = 384;
    float  lightStrideMult = 3.0f;  // shadow rays march 
    int    maxLightSteps   = 48;
};

void initDensityTexture(int3 res);
void uploadDensity(const float* d_linearDensity);
void launchRender(uchar4* d_out, const void* d_vdbGrid, bool useVDB, const RenderParams& rp);
void destroyRenderer();
