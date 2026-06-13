#pragma once
#include <cuda_runtime.h>

struct FluidParams {
    float buoyancy    = 1.4f;   
    float weight      = 0.05f;  
    float vorticity   = 6.0f;   
    float dissipation = 0.998f; 
    float velScale    = 18.0f;  
    int   jacobiIters = 28;     
};

void initFluidSim(int3 res);
void simulateNavierStokes(float dt, float simTime, const FluidParams& fp);
const float* fluidDensityField();   
void destroyFluidSim();
