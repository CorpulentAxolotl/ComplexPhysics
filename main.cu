#include "C:\Code\cpp\physics\particleTracer\glad\glad.h"
#define GLFW_INCLUDE_NONE
#include <C:\Code\cpp\physics\particleTracer\libs\glfw\include\GLFW\glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>
struct Sphere {
    float3 center;
    float radius;
    uchar4 color;
};
__device__ inline float3 make_vec3(float x, float y, float z) {
    return make_float3(x, y, z);
}

__device__ inline float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline float3 operator*(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ inline float3 operator*(float s, const float3& a) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ inline bool uchar4_eq(const uchar4& a, const uchar4& b) {
    return (a.w == b.w && a.x == b.x && a.y == b.y && a.z == b.z);
}
__device__ inline float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline float length(const float3& a) {
    return sqrtf(dot(a, a));
}

__device__ inline float3 normalize(const float3& a) {
    float inv = 1.0f / length(a);
    return make_float3(a.x * inv, a.y * inv, a.z * inv);
}
struct Mat3 {
    float m[3][3]; // row-major
};
// multiply matrix * vector
__host__ __device__ inline float3 mul(const Mat3& M, const float3& v) {
    return make_float3(
        M.m[0][0]*v.x + M.m[0][1]*v.y + M.m[0][2]*v.z,
        M.m[1][0]*v.x + M.m[1][1]*v.y + M.m[1][2]*v.z,
        M.m[2][0]*v.x + M.m[2][1]*v.y + M.m[2][2]*v.z
    );
}
__host__ __device__ inline Mat3 mm(const Mat3& A, const Mat3& B) {
    Mat3 R;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            R.m[i][j] = A.m[i][0] * B.m[0][j]
                      + A.m[i][1] * B.m[1][j]
                      + A.m[i][2] * B.m[2][j];
        }
    }
    return R;
}
__host__ __device__ inline Mat3 operator* (const Mat3& A, const Mat3& B) {
    return mm(A, B);
}
__host__ __device__ inline Mat3 rotationY(float angle) {
    float c = cosf(angle);
    float s = sinf(angle);
    Mat3 R = {{
        { c, 0, s },
        { 0, 1, 0 },
        { -s, 0, c }
    }};
    return R;
}
__host__ __device__ inline Mat3 rotation(float pitch, float roll) {
    float c1 = cosf(pitch);
    float s1 = sinf(pitch);
    float c2 = cosf(roll);
    float s2 = sinf(roll);
    Mat3 Rpitch = {{
        { c1, 0, s1},
        { 0,  1, 0 },
        { -s1,0, c1}
    }};
    Mat3 Rroll = {{
        { 1, 0,   0 },
        { 0, c2, -s2},
        { 0, s2,  c2}
    }};
    return Rpitch*Rroll;
}
__global__ void renderKernel(uchar4* pixels, uchar4* frameBuffer, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;

    pixels[idx] = frameBuffer[idx];
}
__global__ void rayTraceKernel(uchar4* frameBuffer, float3* PosFrame, float3* velFrame, Sphere* spheres, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;

    // compute normalized ray direction
    float u = ((x / (float)width) * 2.0f - 1.0f)*(((float) width)/height);
    float v = (y / (float)height) * 2.0f - 1.0f;


    velFrame[idx] = normalize(make_float3(u, v, -0.8f));
    PosFrame[idx] = PosFrame[idx] + velFrame[idx]*0.1;
    bool colorq = false;
    for (int i = 0; i < 3; ++i) {
        Sphere s = spheres[i];

        float3 oc = PosFrame[idx] - s.center;
        if (length(oc) <= s.radius) {
            if (uchar4_eq(frameBuffer[idx], make_uchar4(0,0,0,0))) {
                frameBuffer[idx] = s.color;
            }
        }
    }
}
// simple shader sources
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

// util: compile shader
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

// globals
GLuint pbo, tex, shaderProg, vao;
cudaGraphicsResource* cudaResource;
int width = 800, height = 600;

uchar4* devFrameBuffer = nullptr;
float3* devParticles;
float3* devVelocities;
float3 cameraPos = make_float3(0, 0, 1.5f);
static float cameraPitch = 0.0f;
static float cameraRoll = 0.0f;
static double lastX = width / 2.0;
static double lastY = height / 2.0;
bool cursorLocked = true;



void initGLObjects() {
    // fullscreen quad
    float quadVertices[] = {
        // positions   // texcoords
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

    // shader program
    GLuint vs = compileShader(GL_VERTEX_SHADER, vertexShaderSrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSrc);
    shaderProg = glCreateProgram();
    glAttachShader(shaderProg, vs);
    glAttachShader(shaderProg, fs);
    glLinkProgram(shaderProg);
    glDeleteShader(vs);
    glDeleteShader(fs);

    // texture
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    // pixel buffer (PBO)
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // register with CUDA
    cudaGraphicsGLRegisterBuffer(&cudaResource, pbo, cudaGraphicsRegisterFlagsWriteDiscard);
}

void renderFrame(float3 camPos, float camRotY, float camRotX, Sphere* devS) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    //ratrace
    rayTraceKernel<<<grid, block>>>(devFrameBuffer, devParticles, devVelocities, devS, width, height);
    cudaDeviceSynchronize();

    // map CUDA resource
    uchar4* devPtr;
    size_t size;
    cudaGraphicsMapResources(1, &cudaResource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, cudaResource);

    renderKernel<<<grid, block>>>(devPtr, devFrameBuffer, width, height);
    cudaDeviceSynchronize();

    cudaGraphicsUnmapResources(1, &cudaResource, 0);

    // upload PBO to texture
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, 
        GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // draw quad
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(shaderProg);
    glBindVertexArray(vao);
    glBindTexture(GL_TEXTURE_2D, tex);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

int main() {
    if (!glfwInit()) return -1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(width, height, "CUDA Raytracer", NULL, NULL);
    if (!window) {glfwTerminate();return -1;}
    glfwMakeContextCurrent(window);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to init GLAD\n";
        return -1;
    }

    initGLObjects();

    cudaMalloc(&devFrameBuffer, width * height * sizeof(uchar4));
    cudaMemset(devFrameBuffer, 0, width * height * sizeof(uchar4));

    cudaMalloc(&devParticles, width * height * sizeof(float3));
    float3* hostParticles = new float3[width*height];
    for (int i = 0; i < width*height; i++) {
        hostParticles[i] = make_float3(0.0f, 0.0f, -1.0f);
    }
    // ... fill with initial positions ...
    cudaMemcpy(devParticles, hostParticles, width*height * sizeof(float3), cudaMemcpyHostToDevice);
    delete[] hostParticles;
    cudaMalloc(&devVelocities, width*height * sizeof(float3));
    float3* hostVel = new float3[width*height];
    for (int i=0;i<width*height;++i) hostVel[i] = make_float3(0,0,-1.0f);

    cudaMemcpy(devVelocities, hostVel, width*height*sizeof(float3), cudaMemcpyHostToDevice);
    delete[] hostVel;


    Sphere spheres[3];
    spheres[0] = {make_float3(0,0,0), 0.5f, make_uchar4(255,0,0,255)};
    spheres[1] = {make_float3(1,0,0), 0.3f, make_uchar4(0,255,0,255)};
    spheres[2] = {make_float3(-1,0,0), 0.7f, make_uchar4(0,0,255,255)};

    Sphere* devSpheres;
    cudaMalloc(&devSpheres, sizeof(spheres));

    cudaMemcpy(devSpheres, spheres, sizeof(spheres), cudaMemcpyHostToDevice);
    while (!glfwWindowShouldClose(window)) {
        // --- input ---
        /* if (cursorLocked && false) {

            if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
                Mat3 rot = rotationY(cameraPitch);
                float3 delta = mul(rot, make_float3(0, 0, -0.1f));
                cameraPos.x += delta.x;
                cameraPos.y += delta.y;
                cameraPos.z += delta.z;
            }
            if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
                Mat3 rot = rotationY(cameraPitch);
                float3 delta = mul(rot, make_float3(0, 0, 0.1f));
                cameraPos.x += delta.x;
                cameraPos.y += delta.y;
                cameraPos.z += delta.z;
            }
            if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
                Mat3 rot = rotationY(cameraPitch);
                float3 delta = mul(rot, make_float3(-0.1f, 0, 0));
                cameraPos.x += delta.x;
                cameraPos.y += delta.y;
                cameraPos.z += delta.z;
            }
            if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
                Mat3 rot = rotationY(cameraPitch);
                float3 delta = mul(rot, make_float3(0.1f, 0, 0));
                cameraPos.x += delta.x;
                cameraPos.y += delta.y;
                cameraPos.z += delta.z;
            }
            if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
                cameraPos.y += 0.1f;
            }
            if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
                cameraPos.y -= 0.1f;
            }
            double mouseX, mouseY;
            glfwGetCursorPos(window, &mouseX, &mouseY);
            double deltaX = mouseX - lastX;
            double deltaY = mouseY - lastY;
            lastX = mouseX;
            lastY = mouseY;
            float sensitivity = 0.002f; 
            cameraPitch -= (float)deltaX * sensitivity;
            cameraRoll -= (float)deltaY * sensitivity;
        }
            */
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        

        renderFrame(cameraPos, cameraPitch, cameraRoll, devSpheres);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    cudaFree(devFrameBuffer);
    return 0;
}