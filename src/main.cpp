#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <fstream>
#include <vector>
#include <chrono>

#include "renderer.h"
#include "fluid_sim.h"

#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), __FILE__, __LINE__); \
        return 1; \
    } } while (0)

static const float kCamToWorld[16] = {
    0.844328f, 0.f, -0.535827f, 0.f,
   -0.170907f, 0.947768f, -0.269306f, 0.f,
    0.50784f,  0.318959f,  0.800227f, 0.f,
    83.292171f, 45.137326f, 126.430772f, 1.f
};

int main(int argc, char** argv)
{
    const int W = 960, H = 540;
    const int SIM_RES = 128;
    int numFrames = (argc > 1) ? atoi(argv[1]) : 120;
    const float dt = 1.0f / 30.0f;

    int3 simRes = make_int3(SIM_RES, SIM_RES, SIM_RES);
    initFluidSim(simRes);
    initDensityTexture(simRes);
    FluidParams fp;

    RenderParams rp = {};
    rp.width = W; rp.height = H;
    rp.aspect = W / (float)H;
    rp.focal = tanf(3.14159265f / 180.0f * 45.0f * 0.5f);
    memcpy(rp.c2w, kCamToWorld, sizeof(kCamToWorld));
    rp.bmin = make_float3(-30, -30, -30);
    rp.bmax = make_float3( 30,  30,  30);
    rp.lightDir   = make_float3(-0.315798f, 0.719361f, 0.618702f);
    rp.lightColor = make_float3(20, 19, 17.5f);
    rp.background = make_float3(0.572f, 0.772f, 0.921f);

    uchar4* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)W * H * sizeof(uchar4)));
    std::vector<uchar4> h_out((size_t)W * H);
    std::vector<unsigned char> rgb((size_t)W * H * 3);

    // we just the plume develop a bit before frame 0
    float simTime = 0.0f;
    for (int i = 0; i < 60; ++i) { simulateNavierStokes(dt, simTime, fp); simTime += dt; }

    auto t0 = std::chrono::steady_clock::now();
    for (int frame = 0; frame < numFrames; ++frame) {
        simulateNavierStokes(dt, simTime, fp);
        simTime += dt;
        uploadDensity(fluidDensityField());
        launchRender(d_out, nullptr, false, rp);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, (size_t)W * H * sizeof(uchar4),
                              cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < (size_t)W * H; ++i) {
            rgb[i*3+0] = h_out[i].x; rgb[i*3+1] = h_out[i].y; rgb[i*3+2] = h_out[i].z;
        }
        char name[64];
        snprintf(name, sizeof(name), "frame.%04d.ppm", frame);
        std::ofstream ofs(name, std::ios::binary);
        ofs << "P6\n" << W << " " << H << "\n255\n";
        ofs.write((const char*)rgb.data(), rgb.size());
        fprintf(stderr, "\rframe %d/%d", frame + 1, numFrames);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    fprintf(stderr, "\n%d frames in %.2fs  (%.1f fps incl. disk I/O)\n",
            numFrames, secs, numFrames / secs);

    cudaFree(d_out);
    destroyRenderer();
    destroyFluidSim();
    return 0;
}
