#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <fstream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cmath>

#include "kernal.cu"

int main()
{
    glfwInit();
    GLFWwindow* window = glfwCreateWindow(width, height, "Volume", NULL, NULL);
    glfwMakeContextCurrent(window);
    glewInit();

    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, width, height, 0,
                 GL_RGB, GL_FLOAT, framebuffer.data());

    while (!glfwWindowShouldClose(window)) 
    {
        glClear(GL_COLOR_BUFFER_BIT);

        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, tex);
        glBegin(GL_QUADS);
            glTexCoord2f(0,0); glVertex2f(-1,-1);
            glTexCoord2f(1,0); glVertex2f( 1,-1);
            glTexCoord2f(1,1); glVertex2f( 1, 1);
            glTexCoord2f(0,1); glVertex2f(-1, 1);
        glEnd();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
}