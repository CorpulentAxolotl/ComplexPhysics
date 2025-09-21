@echo off
setlocal

:: --- Set paths ---

:: --- Compile with nvcc ---
nvcc ^
  -ccbin "cl.exe" ^
  -Xcompiler "/MD" ^
  -I"C:\libs\glad\include" ^
  -I"C:\libs\glfw\include" ^
  -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\include" ^
  maingrok.cu glad.c^
  -L"C:\libs\glfw\lib-vc2022" ^
  -lglfw3 -lopengl32 -luser32 -lgdi32 -lshell32 ^
  -o particleTracer.exe

:: --- Check for errors ---
if %errorlevel% neq 0 (
    echo Build failed! See build.log
    pause
    exit /b %errorlevel%
)

:: --- Run the program ---
particleTracer.exe
pause
