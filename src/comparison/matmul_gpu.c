// matrix_mul_opencl.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <windows.h>

// OpenCL includes - try multiple possible paths
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
// Try standard location first
#if defined(_WIN32)
    // On Windows, check various possible OpenCL SDK locations
    #if defined(CL_SDK_INCLUDE_KHRONOS)
        #include <CL/cl.h>
    #elif defined(NVIDIA_GPU_COMPUTING_SDK)
        #include <CL/cl.h>
    #elif defined(AMD_SDK)
        #include <CL/cl.h>
    #elif defined(INTEL_SDK)
        #include <CL/cl.h>
    #else
        // If no SDK is defined, try common locations
        #if __has_include(<CL/cl.h>)
            #include <CL/cl.h>
        #elif __has_include(<OpenCL/cl.h>)
            #include <OpenCL/cl.h>
        #else
            #define CL_TARGET_OPENCL_VERSION 120
            #include "CL/opencl.h" // Try relative path for downloaded headers
        #endif
    #endif
#else
    // On other platforms
    #include <CL/cl.h>
#endif
#endif

// Function to get time in milliseconds using Windows API
double get_time_ms() {
    LARGE_INTEGER frequency;
    LARGE_INTEGER counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1000.0 / (double)frequency.QuadPart;
}

// Function to check OpenCL errors
void check_error(cl_int err, const char *operation) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error during operation '%s': %d\n", operation, err);
        exit(EXIT_FAILURE);
    }
}

int main() {
    const int N = 32;
    const int SIZE = N * N;
    
    // Start overall timing
    double overall_start = get_time_ms();
    
    // Host data
    float *a_data = (float*)malloc(SIZE * sizeof(float));
    float *b_data = (float*)malloc(SIZE * sizeof(float));
    float *c_data = (float*)malloc(SIZE * sizeof(float));
    
    // Initialize matrices with the same values as the Rust version
    for (int i = 0; i < SIZE; i++) {
        a_data[i] = (float)i;
        b_data[i] = (float)i;
        c_data[i] = 0.0f;
    }
    
    // OpenCL variables
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem a_buffer, b_buffer, c_buffer;
    cl_int err;
    
    // OpenCL kernel - same as in the Rust version
    const char *kernel_src = 
        "__kernel void matmul(\n"
        "    __global const float *a,\n"
        "    __global const float *b,\n"
        "    __global float *c,\n"
        "    const int N\n"
        ") {\n"
        "    int row = get_global_id(0);\n"
        "    int col = get_global_id(1);\n"
        "    float sum = 0.0f;\n"
        "    for (int k=0; k<N; ++k) {\n"
        "        sum += a[row * N + k] * b[k * N + col];\n"
        "    }\n"
        "    c[row * N + col] = sum;\n"
        "}\n";
    
    // Setting up OpenCL computation pipeline
    double setup_start = get_time_ms();
    
    // Get platform and device
    err = clGetPlatformIDs(1, &platform, NULL);
    check_error(err, "Getting platform ID");
    
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    check_error(err, "Getting device ID");
    
    // Create context and command queue
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    check_error(err, "Creating context");
    
    queue = clCreateCommandQueue(context, device, 0, &err);
    check_error(err, "Creating command queue");
    
    // Create program from source
    program = clCreateProgramWithSource(context, 1, &kernel_src, NULL, &err);
    check_error(err, "Creating program");
    
    // Build program
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Error building program: %s\n", log);
        free(log);
        exit(EXIT_FAILURE);
    }
    
    // Create kernel
    kernel = clCreateKernel(program, "matmul", &err);
    check_error(err, "Creating kernel");
    
    double setup_time = get_time_ms() - setup_start;
    printf("OpenCL setup time: %.3f ms\n", setup_time);
    
    // Create buffers
    a_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, SIZE * sizeof(float), NULL, &err);
    check_error(err, "Creating buffer a_buffer");
    
    b_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, SIZE * sizeof(float), NULL, &err);
    check_error(err, "Creating buffer b_buffer");
    
    c_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, SIZE * sizeof(float), NULL, &err);
    check_error(err, "Creating buffer c_buffer");
    
    // Write data to GPU buffers
    double write_start = get_time_ms();
    
    err = clEnqueueWriteBuffer(queue, a_buffer, CL_TRUE, 0, SIZE * sizeof(float), a_data, 0, NULL, NULL);
    check_error(err, "Writing to a_buffer");
    
    err = clEnqueueWriteBuffer(queue, b_buffer, CL_TRUE, 0, SIZE * sizeof(float), b_data, 0, NULL, NULL);
    check_error(err, "Writing to b_buffer");
    
    double write_time = get_time_ms() - write_start;
    printf("Buffer write time: %.3f ms\n", write_time);
    
    // Building and setting up the kernel
    double kernel_build_start = get_time_ms();
    
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_buffer);
    check_error(err, "Setting kernel arg 0");
    
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_buffer);
    check_error(err, "Setting kernel arg 1");
    
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_buffer);
    check_error(err, "Setting kernel arg 2");
    
    int n_val = N;
    err = clSetKernelArg(kernel, 3, sizeof(int), &n_val);
    check_error(err, "Setting kernel arg 3");
    
    double kernel_build_time = get_time_ms() - kernel_build_start;
    printf("Kernel build time: %.3f ms\n", kernel_build_time);
    
    // Execute the kernel
    size_t global_work_size[2] = { N, N };
    
    double execution_start = get_time_ms();
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
    check_error(err, "Enqueueing kernel");
    
    // Wait for execution to finish
    clFinish(queue);
    double execution_time = get_time_ms() - execution_start;
    printf("Kernel execution time: %.3f ms\n", execution_time);
    
    // Read result back
    double read_start = get_time_ms();
    err = clEnqueueReadBuffer(queue, c_buffer, CL_TRUE, 0, SIZE * sizeof(float), c_data, 0, NULL, NULL);
    check_error(err, "Reading from c_buffer");
    double read_time = get_time_ms() - read_start;
    printf("Buffer read time: %.3f ms\n", read_time);
    
    // Verify output
    int errors = 0;
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            float expected = 0.0f;
            for (int k = 0; k < N; k++) {
                expected += a_data[row * N + k] * b_data[k * N + col];
            }
            
            int idx = row * N + col;
            if (fabs(c_data[idx] - expected) > 1e-3) {
                errors++;
            }
        }
    }
    
    if (errors == 0) {
        printf("Matmul done. All results correct.\n");
    } else {
        printf("Matmul done with %d errors.\n", errors);
    }
    
    // Log total time
    double total_time = get_time_ms() - overall_start;
    printf("Total execution time: %.3f ms\n", total_time);
    
    // Clean up
    clReleaseMemObject(a_buffer);
    clReleaseMemObject(b_buffer);
    clReleaseMemObject(c_buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    free(a_data);
    free(b_data);
    free(c_data);
    
    return 0;
}