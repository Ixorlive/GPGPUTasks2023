#include <CL/cl.h>
#include <libclew/ocl_init.h>
#include <libutils/fast_random.h>
#include <libutils/timer.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <sys/types.h>
#include <vector>


template<typename T>
std::string to_string(T value) {
    std::ostringstream ss;
    ss << value;
    return ss.str();
}

void reportError(cl_int err, const std::string &filename, int line) {
    if (CL_SUCCESS == err)
        return;

    std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
    throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)

cl_device_id selectDevice(bool prefGPU = true) {
    cl_uint platformsCount = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
    std::vector<cl_platform_id> platforms(platformsCount);
    OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

    cl_device_id best_device_id = nullptr;

    size_t nameSize;
    std::vector<char> nameData;

    for (const auto &platform : platforms) {
        cl_uint devicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
        if (devicesCount == 0)
            continue;

        std::vector<cl_device_id> devices(devicesCount);
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

        for (const auto &device : devices) {
            cl_device_type deviceType;

            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &deviceType, nullptr));
            OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &nameSize));
            nameData.resize(nameSize);
            OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, nameSize, nameData.data(), nullptr));

            std::string vendor(nameData.begin(), nameData.end());

            //sorry I just want to choose NVIDIA instead of Intel (I have dual GPU)
            if (vendor.find("NVIDIA") != std::string::npos) {
                return device;
            } else if ((deviceType & CL_DEVICE_TYPE_GPU && prefGPU) || !prefGPU) {
                best_device_id = device;
            }
        }
    }

    if (best_device_id == nullptr) {
        throw std::runtime_error("No suitable OpenCL device found.");
    }

    return best_device_id;
}

int main() {
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    // Select OpenCL device (choose gpu as prefable, in my case I want to choose NVIDIA)
    cl_device_id device = selectDevice();
    // Print selected OpenCL device
    char buffer[1024];
    OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(buffer), buffer, nullptr));
    std::cout << "Seleceted device name: " << buffer << std::endl;

    cl_int errcode_ret;
    // Create OpenCL context
    cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);

    // Create Command Queue
    auto command_queue = clCreateCommandQueue(ctx, device, 0, &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);

    unsigned int n = 1000 * 1000 * 100;
    // Создаем два массива псевдослучайных данных для сложения и массив для будущего хранения результата
    std::vector<float> as(n, 0);
    std::vector<float> bs(n, 0);
    std::vector<float> cs(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.nextf();
        bs[i] = r.nextf();
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    // Create OpenCL buffers
    cl_mem as_buffer =
            clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * n, as.data(), &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);
    cl_mem bs_buffer =
            clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * n, bs.data(), &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);
    cl_mem cs_buffer = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(float) * n, nullptr, &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);

    std::string kernel_sources;
    {
        std::ifstream file("src/cl/aplusb.cl");
        kernel_sources = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        if (kernel_sources.size() == 0) {
            throw std::runtime_error("Empty source file! May be you forgot to configure working directory properly?");
        }
    }

    // Create program
    const char *c_kernel_sources = kernel_sources.c_str();
    size_t source_length = kernel_sources.length();
    cl_program program = clCreateProgramWithSource(ctx, 1, &c_kernel_sources, &source_length, &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);
    // Build program
    OCL_SAFE_CALL(clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr));
    // Print logs
    size_t log_size;
    OCL_SAFE_CALL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size));
    std::vector<char> log(log_size);
    OCL_SAFE_CALL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr));

    if (log_size > 1) {
        std::cout << "Log:" << std::endl;
        std::cout << log.data() << std::endl;
    }

    // Create OpenCL kernel
    const char *function_name = "aplusb";
    cl_kernel kernel = clCreateKernel(program, function_name, &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);

    // Set Kernel Arguments and Execute Kernel
    {
        unsigned int i = 0;
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(cl_mem), &as_buffer));
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(cl_mem), &bs_buffer));
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(cl_mem), &cs_buffer));
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(unsigned int), &n));
    }
    // Measure kernel execution time and compute metrics
    {
        size_t workGroupSize = 128;
        size_t global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        timer t;
        for (unsigned int i = 0; i < 20; ++i) {
            cl_event event;
            OCL_SAFE_CALL(clEnqueueNDRangeKernel(command_queue, kernel, 1, nullptr, &global_work_size, &workGroupSize,
                                                 0, nullptr, &event));
            OCL_SAFE_CALL(clWaitForEvents(1, &event));
            OCL_SAFE_CALL(clReleaseEvent(event));
            t.nextLap();
        }
        std::cout << "Kernel average time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;

        std::cout << "GFlops: " << (n / t.lapAvg()) * 1e-9 << std::endl;

        double total_data_gb = static_cast<double>(3 * n * sizeof(float)) / (1024.0 * 1024.0 * 1024.0);// in GB
        double avg_time = t.lapAvg();                                                                  // in seconds

        double bandwidth = total_data_gb / avg_time;// in GB/s

        std::cout << "VRAM bandwidth: " << bandwidth << " GB/s" << std::endl;
    }
    // Measure data transfer time from VRAM to RAM and compute bandwidth
    {
        timer t;
        for (unsigned int i = 0; i < 20; ++i) {
            cl_event event;
            OCL_SAFE_CALL(clEnqueueReadBuffer(command_queue, cs_buffer, CL_TRUE, 0, n * sizeof(float), cs.data(), 0,
                                              nullptr, &event));
            OCL_SAFE_CALL(clWaitForEvents(1, &event));
            OCL_SAFE_CALL(clReleaseEvent(event));
            t.nextLap();
        }
        std::cout << "Result data transfer time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        double avgTime = t.lapAvg();                                   // Average time in seconds
        double dataSize = static_cast<double>(n * sizeof(float));      // Size of data in bytes
        double bandwidth = (dataSize / avgTime) / (1024 * 1024 * 1024);// Bandwidth in GB/s

        std::cout << "VRAM -> RAM bandwidth: " << bandwidth << " GB/s" << std::endl;
    }

    float eps = 1e-6;
    for (unsigned int i = 0; i < n; ++i) {
        float rel_error = std::abs(cs[i] - as[i] - bs[i]) / std::max(std::abs(cs[i]), std::abs(as[i] + bs[i]));
        if (rel_error > eps) {
            std::cout << "cs[i]: " << cs[i] << ", as[i]: " << as[i] << ", bs[i]: " << bs[i] << std::endl;
            throw std::runtime_error("CPU and GPU results differ!");
        }
    }

    // Cleanup (TODO: RAII)
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseMemObject(as_buffer);
    clReleaseMemObject(bs_buffer);
    clReleaseMemObject(cs_buffer);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(ctx);

    return 0;
}
