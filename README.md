# Volumetric Daddy


## Project Over view

A Real-Time Volumetric Renderer optimized to work on the GPU. with Smoke being generated from Navier-Stokes equations specifically Stam Style Stable Fluids.

## How the GPU is Being Used

### Volumetric Renderer

The Renderer has it's pixels parallized. To determine the color of each pixel it is computed through an independant ray marcher. Each pixel shoots an imaginary ray from the camera and towards the smoke. As it reaches the smoke it steps through the volume, accumalating the amount of light that is blocked and scattered. Since each pixel is extremely independent we can compute all 960x540 pixels simultaneously.
```cpp
dim3 block(16, 16);
dim3 grid((rp.width + 15) / 16, (rp.height + 15) / 16);

if (i >= rp.width || j >= rp.height) return; // needs to be ran since it is upperbounded

cppint i = blockIdx.x * blockDim.x + threadIdx.x;
int j = blockIdx.y * blockDim.y + threadIdx.y;
```
The only thing that is parallel and is handled well by the GPU is that each pixel needs to be calculated independantly. However, past that the rest are severely sequential. 300 steps front to back through the volume, with hose step must happen in order, due to light accumulating as it travels. 

### Stam Stable Style  (Navier-Stokes Fluids)

As for the Stam Style Stable fluids this is parallized since the values of the smoke are stored in a 128x128x128 cell cube. Since each cell could update with just the values of themselves and it's immediate neighbors(up, down, left, right, forwards, backwards), They could be updated at the same time 

```cpp
dim3 b(8, 8, 8);
dim3 g((g_res.x + 7) / 8, (g_res.y + 7) / 8, (g_res.z + 7) / 8);

if (x >= r.x || y >= r.y || z >= r.z)
```

As for the stam Style Stable Fluids the 3D volume: one thread per cell, 2 million cells (128³) all updated at once, within each stage. Since the pixels are not fully independant each cell eads its neighbors to compute its update. If a cell overwrote its value while a neighbor was still reading the old one, the result would be corrupted and depend on random timing. The fix is double buffering: read from the "before" grid, write to a separate "after" grid, then swap. Now every cell reads a consistent frozen snapshot, so all 2 million updates can safely run in parallel despite the neighbor dependencies.

what does stays serial is calculating Advection -> buonyancy -> vorticity -> pressure -> density, followed by the 28 iterations of the pressure solver.

## Detailed implementation:

### Volumetric Renderer

The renderer is a single CUDA kernel where each thread handles one pixel of the 960x540 image, launched as a 2D grid of 16x16 blocks. Each thread starts by building its camera ray, transforming a pixel direction through the camera-to-world matrix, then runs the ray-box intersection test to see if that ray even hits the volume. This is the same slab-method test from the original CPU code, just moved onto the device. If the ray misses, the thread is done. If it hits, the thread marches through the volume in fixed-size steps from the entry point to the exit point. At each step it reads the smoke density out of a 3D texture, which is the part I'm happiest with, since setting the texture to linear filtering means the GPU's texture hardware does the trilinear interpolation for free instead of me computing eight weighted reads by hand like the CPU version does. Every sample dims the running transmittance using Beer-Lambert and adds in whatever light is scattering toward the camera, weighted by the Henyey-Greenstein phase function. There's also a second smaller march from each sample toward the light to get self-shadowing, but I run that one at a coarser step with a hard cap on how many steps it takes so it doesn't tank the frame rate. The march bails out early once the ray is basically opaque, which replaces the random Russian-roulette termination from the CPU code (that relied on a random call that doesn't exist on the GPU). The final color gets tonemapped and written straight into an OpenGL buffer so the frame never has to leave the card.

### Stam's Stable Style Fluids

The fluid solver stores the smoke as a dense 128-cubed grid, with separate arrays on the GPU for velocity, density, pressure, and a few scratch buffers. Each step of the simulation is its own kernel, launched over a 3D grid of 8x8x8 blocks with one thread per cell, and each thread reads its six neighbors through a helper that clamps at the grid edges so nothing reads out of bounds. A full timestep just runs the stages in order: first advection, where each cell traces backward along the velocity field and resamples there, which is the trick that keeps the whole thing stable no matter how big the timestep is. Then a buoyancy push so the smoke rises. Then the curl and vorticity-confinement steps that add the swirling motion back in, since advection tends to smooth it out. After that comes the pressure projection, which is what forces the fluid to actually behave like a fluid, and it's the MOST expensive part: it computes the divergence, runs 28 rounds of the same Jacobi smoothing kernel, then subtracts the result back out of the velocity. Every kernel reads its neighbors, so they all write into a second copy of the grid and the host swaps the two afterward, which keeps threads from overwriting values their neighbors still need. The stages run one after another and those 28 pressure rounds have to go in sequence, but inside any single kernel all two million cells update at the same time. Whatever density falls out at the end gets copied straight into the renderer's texture on the GPU, so nothing ever round-trips back to the CPU.

## How To Run It
This project runs on UCR's Bender GPU server inside the course Apptainer image.

### 1. Enter the Apptainer environment

The `--nv` flag binds the GPU into the image and is required.

```bash
apptainer shell --nv /scratch/csee147/csee147env.sif
```

### 2. Build

```bash
make
```

This compiles the CUDA renderer and fluid simulation into an executable called `render`.

### 3. Render the frames

```bash
make run
```

This runs the simulation and renderer, writing each frame to disk as `frame.0000.ppm`, `frame.0001.ppm`, and so on. By default it renders 120 frames; to choose a different count, run `./render` directly with a number, for example `./render 300`.

### 4. Encode the frames into a video

```bash
make video
```

This stitches the frames into `smoke.mp4`. If `ffmpeg` is not available, install it once with `pip install imageio-ffmpeg` inside the shell and point the encode at the bundled binary.

### 5. Clean up

```bash
make clean
```

This removes the executable, the rendered frames But keeps the video.


## Results

## Presenation of my work

## Problems faced

The problem I faced the most is reading the Stable Fluids paper by Jos Stam. Using AI to help me interpret the text help me understand what I needed to do when writing the code for cloud simulator.

The other Problem I faced is that I could not use openGL on blender. Since blender does not have a monitor nor display server bound to it's GPU I am unable to make this truly realtime. In the end I had to generate each frame and compile them into a video using ffmpeg.

| Task | BreakDown |
|----------|----------|
| everything    | Oscar 100%     |