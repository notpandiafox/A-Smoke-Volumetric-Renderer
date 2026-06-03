Libraries and Framwork needed:
  1. CUDA ToolKit
  2. GFLW
  3. GLAD/GLED
  4. OpenGL
  5. GLM (OpenGL Mathematics)
  6. CUDA Thrust
  7. Dear ImGui
  8. nanoVDB

Issues that will be encountered:
  1. One issue that I will encounter is the setup, making sure that the CUDA toolkit version, compiler and the blender GPU will all be compatitable.
  2. Understanding how to build a render from scratch such as how rays are generated from the camera, how rays intersects a volume boundary.
  3. I feel that since the GPU is harder to debug debugging will be a huge issue when making it. Especially if I get a ray wrong the renderer would produce nothing.

Outline:
  1. Read ScratchPixel's Volume rendering introduction 
    https://www.scratchapixel.com/lessons/3d-basic-rendering/volume-rendering-for-developers/volume-rendering-summary-equations.html
  2. take a look and understnad the CUDA's volume Renderer to understand the structure
  3. get the sample volume renderer to work first to understand how it is suppose to look like
  4. then get a window up and running with GLFW
  5. create a color gradient on the screen.
  6. get the camera to work and ray generation.
  7. Create a basic ray marcher
  8. create a 3d noise density field
If I have time
  9. create a transfer function
  10. add directional lighting with shadow rays
  11. add ImGui controlers such as sliders light intensity, light transfer step size and the transfer function 
