#include <iostream>
#include <cmath>
#include <vector_types.h>
#include <vector_functions.h>

#include "C:\\Code\\cpp\\physics\\particleTracer\\glad\\glad.h"  // Adjust path as needed
#define GLFW_INCLUDE_NONE
#include "C:\\Code\\cpp\\physics\\particleTracer\\libs\\glfw\\include\\GLFW\\glfw3.h"  // Adjust path as needed
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

struct Sphere {
    float3 center;
    float radius;
    uchar4 color;
};

__device__ inline float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline float3 operator*(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__host__ __device__ inline float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline float length(const float3& a) {
    return sqrtf(dot(a, a));
}

__host__ __device__ inline float3 normalize(const float3& a) {
    float inv = 1.0f / length(a);
    return make_float3(a.x * inv, a.y * inv, a.z * inv);
}

__host__ __device__ inline bool uchar4_eq(const uchar4& a, const uchar4& b) {
    return (a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w);
}

struct Mat3 {
    float m[3][3]; // row-major
};

__host__ __device__ inline float3 mul(const Mat3& M, const float3& v) {
    return make_float3(
        M.m[0][0]*v.x + M.m[0][1]*v.y + M.m[0][2]*v.z,
        M.m[1][0]*v.x + M.m[1][1]*v.y + M.m[1][2]*v.z,
        M.m[2][0]*v.x + M.m[2][1]*v.y + M.m[2][2]*v.z
    );
}

__host__ __device__ inline Mat3 rotation(float pitch, float yaw) {
    float c1 = cosf(pitch);
    float s1 = sinf(pitch);
    float c2 = cosf(yaw);
    float s2 = sinf(yaw);
    Mat3 Rpitch = {{
        { c1, 0, s1 },
        { 0, 1, 0 },
        { -s1, 0, c1 }
    }};
    Mat3 Ryaw = {{
        { c2, 0, s2 },
        { 0, 1, 0 },
        { -s2, 0, c2 }
    }};
    Mat3 R;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            R.m[i][j] = Rpitch.m[i][0] * Ryaw.m[0][j] +
                        Rpitch.m[i][1] * Ryaw.m[1][j] +
                        Rpitch.m[i][2] * Ryaw.m[2][j];
        }
    }
    return R;
}
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>


__host__ __device__ inline
bool pointInsideY(const float3& p, const float3& center,
                  float R2, float halfHeight) {
    float dx = p.x - center.x;
    float dz = p.z - center.z;
    float dy = p.y - center.y;
    return (dx*dx + dz*dz <= R2 + 1e-6f &&
            dy >= -halfHeight - 1e-6f && dy <= halfHeight + 1e-6f);
}

__host__ __device__
bool intersectsWasher(
    float3 p0, float3 p1,       // segment endpoints
    float3 center, float3 dir,  // axis (normalized!)
    float rInner, float rOuter, // inner & outer radii
    float height                // full cylinder height
) {
    float3 n = make_float3(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z); // segment vector
    float3 m = make_float3(p0.x - center.x, p0.y - center.y, p0.z - center.z);

    float md = m.x*dir.x + m.y*dir.y + m.z*dir.z;
    float nd = n.x*dir.x + n.y*dir.y + n.z*dir.z;
    float nn = n.x*n.x + n.y*n.y + n.z*n.z;
    float mn = m.x*n.x + m.y*n.y + m.z*n.z;
    float mm = m.x*m.x + m.y*m.y + m.z*m.z;

    // helper lambda to check height range
    auto inHeight = [&](float axisVal) {
        return (axisVal >= -height*0.5f && axisVal <= height*0.5f);
    };

    // Check side surface of outer cylinder
    {
        float a = nn - nd*nd;
        float b = mn - md*nd;
        float c = mm - md*md - rOuter*rOuter;
        float discr = b*b - a*c;
        if (discr >= 0.0f) {
            float sqrtDiscr = sqrtf(discr);
            float u0 = (-b - sqrtDiscr) / a;
            float u1 = (-b + sqrtDiscr) / a;
            for (float u : {u0, u1}) {
                if (u >= 0.0f && u <= 1.0f) {
                    float hitAxis = md + u*nd;
                    if (inHeight(hitAxis)) {
                        // distance^2 from axis at this hit
                        float dist2 = (mm + 2*u*mn + u*u*nn) - (hitAxis*hitAxis);
                        if (dist2 >= rInner*rInner && dist2 <= rOuter*rOuter) {
                            return true;
                        }
                    }
                }
            }
        }
    }

    // Check caps (two annular disks)
    for (int sign = -1; sign <= 1; sign += 2) {
        float cap = sign * height * 0.5f;
        float denom = nd;
        if (fabsf(denom) > 1e-6f) {
            float u = (cap - md) / denom;
            if (u >= 0.0f && u <= 1.0f) {
                float3 hit = make_float3(p0.x + u*n.x, p0.y + u*n.y, p0.z + u*n.z);
                float3 dvec = make_float3(hit.x - center.x - cap*dir.x,
                                          hit.y - center.y - cap*dir.y,
                                          hit.z - center.z - cap*dir.z);
                float dist2 = dvec.x*dvec.x + dvec.y*dvec.y + dvec.z*dvec.z;
                if (dist2 >= rInner*rInner && dist2 <= rOuter*rOuter) {
                    return true;
                }
            }
        }
    }

    return false;
}



const char* vertexShaderSrc = R"(
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTex;
out vec2 TexCoord;
void main() {
    TexCoord = aTex;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)";

const char* fragmentShaderSrc = R"(
#version 330 core
out vec4 FragColor;
in vec2 TexCoord;
uniform sampler2D screenTex;
void main() {
    FragColor = texture(screenTex, TexCoord);
}
)";

GLuint compileShader(GLenum type, const char* src) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, NULL);
    glCompileShader(shader);
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[512];
        glGetShaderInfoLog(shader, 512, NULL, log);
        std::cerr << "Shader error: " << log << std::endl;
    }
    return shader;
}

GLuint pbo, tex, shaderProg, vao;
cudaGraphicsResource* cudaResource;
int width = 800, height = 600;
float3 cameraPos = make_float3(0.0f, 0.0f, 5.0f);
float cameraPitch = 0.0f;
float cameraYaw = 0.0f;
uchar4* devFrameBuffer;
float3* devParticles;
float3* devVelocities;
Sphere* devSpheres;
float dt = 0.01f;  // Step size for particle advancement

static bool wasMousePressed = false;
void initGLObjects() {
    float quadVertices[] = {
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f,
        -1.0f,  1.0f,  0.0f, 1.0f
    };
    unsigned int indices[] = { 0, 1, 2, 2, 3, 0 };

    GLuint vbo, ebo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &ebo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    GLuint vs = compileShader(GL_VERTEX_SHADER, vertexShaderSrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSrc);
    shaderProg = glCreateProgram();
    glAttachShader(shaderProg, vs);
    glAttachShader(shaderProg, fs);
    glLinkProgram(shaderProg);
    glDeleteShader(vs);
    glDeleteShader(fs);

    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    cudaGraphicsGLRegisterBuffer(&cudaResource, pbo, cudaGraphicsRegisterFlagsWriteDiscard);
}

__global__ void particleUpdateKernel(uchar4* frameBuffer, float3* particles, float3* velocities, Sphere* spheres, int width, int height, float dt, int numSpheres) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = y * width + x;
    if (!uchar4_eq(frameBuffer[idx], make_uchar4(0,0,0,255))) return;

    float3 pos = particles[idx];
    float3 vel = velocities[idx];
    float3 gdir = spheres[0].center - particles[idx];
    float fStrength = 0.006;
    float3 gforce = normalize(gdir)*(fStrength/dot(gdir,gdir));
    float3 step = vel * dt;
    float stepLen = length(step);  // Since vel normalized, stepLen = dt

    // Advance particle
    particles[idx] = pos + step;
    float3 cdir = normalize(make_float3(-1,7,-0.5));
    if (intersectsWasher(pos, particles[idx], spheres[0].center, cdir, 4.6f, 5.0f, 0.01f)) {
        frameBuffer[idx] = make_uchar4(200,200,0,255);
    }
    if (intersectsWasher(pos, particles[idx], spheres[0].center, cdir, 4.0f, 4.5f, 0.01f)) {
        frameBuffer[idx] = make_uchar4(250,100,0,255);
    }
    if (intersectsWasher(pos, particles[idx], spheres[0].center, cdir, 3.3f, 3.9f, 0.01f)) {
        frameBuffer[idx] = make_uchar4(255,150,0,255);
    }
    if (intersectsWasher(pos, particles[idx], spheres[0].center, cdir, 2.3f, 3.2f, 0.01f)) {
        frameBuffer[idx] = make_uchar4(200,200,0,255);
    }
    if (intersectsWasher(pos, particles[idx], spheres[0].center, cdir, 0.7f, 2.0f, 0.01f)) {
        frameBuffer[idx] = make_uchar4(250,100,0,255);
    }
    float3 oc = pos - spheres[0].center;  // From old pos
    float a = dot(step, step);
    float b = 2.0f * dot(oc, step);
    float c = dot(oc, oc) - (fStrength/dt + 0.01) * (fStrength/dt + 0.01);
    float disc = b * b - 4 * a * c;
    if (disc > 0.0f) {
        float sqrtDisc = sqrtf(disc);
        float t1 = (-b - sqrtDisc) / (2.0f * a);
        float t2 = (-b + sqrtDisc) / (2.0f * a);
        // Check if any t in [0,1] (fraction along step)
        if ((t1 >= 0.0f && t1 <= 1.0f) || (t2 >= 0.0f && t2 <= 1.0f)) {
            //frameBuffer[idx] = make_uchar4(10,10,10,255);
        }
    }
    // Check for hit during this step with each sphere
    for (int i = 1; i < numSpheres; ++i) {
        Sphere s = spheres[i];
        float3 oc = pos - s.center;  // From old pos
        float a = dot(step, step);
        float b = 2.0f * dot(oc, step);
        float c = dot(oc, oc) - s.radius * s.radius;
        float disc = b * b - 4 * a * c;

        if (disc > 0.0f) {
            float sqrtDisc = sqrtf(disc);
            float t1 = (-b - sqrtDisc) / (2.0f * a);
            float t2 = (-b + sqrtDisc) / (2.0f * a);

            // Check if any t in [0,1] (fraction along step)
            if ((t1 >= 0.0f && t1 <= 1.0f) || (t2 >= 0.0f && t2 <= 1.0f)) {
                //frameBuffer[idx] = s.color;
            }
        }
    }
    
    
    velocities[idx] = velocities[idx] + gforce*(dt/0.01);
    velocities[idx] = normalize(velocities[idx]);
}

__global__ void copyToPboKernel(uchar4* pboPixels, uchar4* frameBuffer, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    pboPixels[idx] = frameBuffer[idx];
}

void renderFrame() {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // Update particles and check hits
    particleUpdateKernel<<<grid, block>>>(devFrameBuffer, devParticles, devVelocities, devSpheres, width, height, dt, 2);
    cudaDeviceSynchronize();

    // Map PBO
    uchar4* devPtr;
    size_t size;
    cudaGraphicsMapResources(1, &cudaResource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, cudaResource);

    // Copy frameBuffer to PBO
    copyToPboKernel<<<grid, block>>>(devPtr, devFrameBuffer, width, height);
    cudaDeviceSynchronize();

    // Unmap
    cudaGraphicsUnmapResources(1, &cudaResource, 0);

    // Upload to texture
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // Draw
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(shaderProg);
    glBindVertexArray(vao);
    glBindTexture(GL_TEXTURE_2D, tex);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}
void restartSimulation() {
    // Initialize host arrays
    float3* hostParticles = new float3[width * height];
    float3* hostVelocities = new float3[width * height];
    uchar4* hostFrame = new uchar4[width * height];

    // Set frame buffer to black with alpha 255
    for (int i = 0; i < width * height; ++i) {
        hostFrame[i] = make_uchar4(0, 0, 0, 255);
    }

    // Initialize particles and velocities
    Mat3 rot = rotation(cameraPitch, cameraYaw);
    float aspect = (float)width / (float)height;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            float u = ((x / (float)width) * 2.0f - 1.0f) * aspect;
            float v = ((y / (float)height) * 2.0f - 1.0f);
            float3 rawDir = normalize(make_float3(u, -v, -1.0f));
            float3 dir = mul(rot, rawDir);
            hostVelocities[idx] = dir;
            hostParticles[idx] = cameraPos;
        }
    }

    // Copy to device
    cudaMemcpy(devFrameBuffer, hostFrame, width * height * sizeof(uchar4), cudaMemcpyHostToDevice);
    cudaMemcpy(devParticles, hostParticles, width * height * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(devVelocities, hostVelocities, width * height * sizeof(float3), cudaMemcpyHostToDevice);

    // Free host memory
    delete[] hostFrame;
    delete[] hostParticles;
    delete[] hostVelocities;

    // Ensure CUDA operations are complete
    cudaDeviceSynchronize();
}
Sphere editSpheres(Sphere* spheres, Sphere* devSpheres, int numSpheres, float x, float y) {
    spheres[1].center.x += x;
    spheres[1].center.y += y;
    cudaMemcpy(devSpheres, spheres, numSpheres * sizeof(Sphere), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    return spheres[2];
}
int main() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(width, height, "Particle Tracer", NULL, NULL);
    if (!window) {
        glfwTerminate();
        std::cerr << "Failed to create GLFW window\n";
        return -1;
    }
    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to init GLAD\n";
        glfwTerminate();
        return -1;
    }

    initGLObjects();

    // Allocate device buffers
    cudaMalloc(&devFrameBuffer, width * height * sizeof(uchar4));
    cudaMemset(devFrameBuffer, 0, width * height * sizeof(uchar4));

    // Set frameBuffer to black with alpha 255
    uchar4* hostFrame = new uchar4[width * height];
    for (int i = 0; i < width * height; ++i) {
        hostFrame[i] = make_uchar4(0, 0, 0, 255);
    }
    cudaMemcpy(devFrameBuffer, hostFrame, width * height * sizeof(uchar4), cudaMemcpyHostToDevice);
    delete[] hostFrame;

    cudaMalloc(&devParticles, width * height * sizeof(float3));
    cudaMalloc(&devVelocities, width * height * sizeof(float3));

    // Initialize particles and velocities on host
    float3* hostParticles = new float3[width * height];
    float3* hostVelocities = new float3[width * height];
    Mat3 rot = rotation(cameraPitch, cameraYaw);
    float aspect = (float)width / (float)height;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            float u = ((x / (float)width) * 2.0f - 1.0f) * aspect;
            float v = ((y / (float)height) * 2.0f - 1.0f);  // Note: v flipped if needed, but ok
            float3 rawDir = normalize(make_float3(u, -v, -1.0f));  // Flip v for right-side up
            float3 dir = mul(rot, rawDir);
            hostVelocities[idx] = dir;
            hostParticles[idx] = cameraPos;
        }
    }

    cudaMemcpy(devParticles, hostParticles, width * height * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(devVelocities, hostVelocities, width * height * sizeof(float3), cudaMemcpyHostToDevice);
    delete[] hostParticles;
    delete[] hostVelocities;

    // Spheres
    Sphere spheres[2];
    spheres[0] = {make_float3(0.0f, 0.0f, -3.0f), 0.8f, make_uchar4(255, 0, 0, 255)};
    spheres[1] = {make_float3(2.0f, 2.0f, -6.0f), 0.2f, make_uchar4(0, 255, 0, 255)};

    cudaMalloc(&devSpheres, 2 * sizeof(Sphere));
    cudaMemcpy(devSpheres, spheres, 2 * sizeof(Sphere), cudaMemcpyHostToDevice);

    while (!glfwWindowShouldClose(window)) {
        
        bool isMousePressed = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
        if (isMousePressed && !wasMousePressed) {
            restartSimulation();
        }
        wasMousePressed = isMousePressed;

        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
            spheres[2] = editSpheres(spheres, devSpheres, 2, -0.1f, 0);
        }
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
            spheres[2] = editSpheres(spheres, devSpheres, 2, 0.1f, 0);
        }
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
            spheres[2] = editSpheres(spheres, devSpheres, 2, 0, -0.1f);
        }
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
            spheres[2] = editSpheres(spheres, devSpheres, 2, 0, 0.1f);
        }

        renderFrame();
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    cudaFree(devFrameBuffer);
    cudaFree(devParticles);
    cudaFree(devVelocities);
    cudaFree(devSpheres);
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &tex);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(shaderProg);
    cudaGraphicsUnregisterResource(cudaResource);
    glfwTerminate();
    return 0;
}