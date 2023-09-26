#include "CL/cl_platform.h"
#include "libgpu/context.h"
#include "libgpu/shared_device_buffer.h"
#include <functional>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

#include "cl/sum_cl.h"
#include "libgpu/work_size.h"

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


int main(int argc, char **argv) {
    int benchmarkingIters = 10;

    unsigned int reference_sum = 0;
    unsigned int n = 100 * 1000 * 1000;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(42);
    for (int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<unsigned int>::max() / n);
        reference_sum += as[i];
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU:     " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
#pragma omp parallel for reduction(+ : sum)
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU OpenMP result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU OMP: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU OMP: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }
    // GPU testing
    const size_t VALUES_PER_WORKITEM = 64;
    const size_t WORKGROUP_SIZE = 64;

    gpu::Device device = gpu::chooseGPUDevice(argc, argv);
    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    gpu::gpu_mem_32u as_gpu;
    gpu::gpu_mem_32u sum_gpu;
    as_gpu.resizeN(n);
    sum_gpu.resizeN(1);
    as_gpu.writeN(as.data(), n);

    auto executeKernel = [&](const std::string &funTestName, const std::function<void(ocl::Kernel &)> &execFunc,
                             const std::string &errMsg, const std::string &logMsg) {
        ocl::Kernel kernel(sum_kernel, sum_kernel_length, funTestName);
        kernel.compile();

        timer t;
        cl_uint sum = 0;
        for (size_t i = 0; i < benchmarkingIters; i++) {
            sum = 0;
            sum_gpu.writeN(&sum, 1);
            execFunc(kernel);
            t.nextLap();
        }
        t.stop();
        sum_gpu.readN(&sum, 1);
        EXPECT_THE_SAME(reference_sum, sum, errMsg);
        std::cout << logMsg << ": " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << logMsg << ": " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    };

    size_t workSize = (n + VALUES_PER_WORKITEM - 1) / VALUES_PER_WORKITEM;

    executeKernel(
            "sum_atomic",
            [&](ocl::Kernel &kernel) {
                kernel.exec(gpu::WorkSize(WORKGROUP_SIZE, n), as_gpu.clmem(), sum_gpu.clmem(), n);
            },
            "GPU atomic sum should be consistent!", "Atomic sum");

    executeKernel(
            "sum_loop",
            [&](ocl::Kernel &kernel) {
                kernel.exec(gpu::WorkSize(WORKGROUP_SIZE, workSize), as_gpu.clmem(), sum_gpu.clmem(), n);
            },
            "GPU loop sum should be consistent!", "Loop sum");

    executeKernel(
            "sum_loop_coalesced",
            [&](ocl::Kernel &kernel) {
                kernel.exec(gpu::WorkSize(WORKGROUP_SIZE, workSize), as_gpu.clmem(), sum_gpu.clmem(), n);
            },
            "GPU loop coalesced sum should be consistent!", "Loop coalesced sum");

    executeKernel(
            "sum_local_mem_thread",
            [&](ocl::Kernel &kernel) {
                kernel.exec(gpu::WorkSize(WORKGROUP_SIZE, n), as_gpu.clmem(), sum_gpu.clmem(), n);
            },
            "GPU local mem thread sum should be consistent!", "Local mem sum");

    executeKernel(
            "sum_tree_local",
            [&](ocl::Kernel &kernel) {
                kernel.exec(gpu::WorkSize(WORKGROUP_SIZE, n), as_gpu.clmem(), sum_gpu.clmem(), n);
            },
            "GPU local tree sum should be consistent!", "Local tree sum");
    {
        // Tree global sum
        const std::string FUN_TEST_NAME = "sum_tree_global";
        ocl::Kernel kernel(sum_kernel, sum_kernel_length, FUN_TEST_NAME);
        kernel.compile();
        gpu::gpu_mem_32u temp_src;
        gpu::gpu_mem_32u temp_dst;

        temp_src.resizeN(n);
        temp_dst.resizeN(n);

        timer t;
        cl_uint sum = 0;
        for (size_t i = 0; i < benchmarkingIters; i++) {
            as_gpu.copyToN(temp_src, n);
            uint work_size = n;
            while (work_size > 1) {
                kernel.exec(gpu::WorkSize(WORKGROUP_SIZE, work_size), temp_src.clmem(), temp_dst.clmem(), work_size);
                work_size = (work_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
                std::swap(temp_src, temp_dst);
            }
            t.nextLap();
        }
        t.stop();
        temp_src.readN(&sum, 1);
        EXPECT_THE_SAME(reference_sum, sum, "GPU tree global sum should be consistent!");
        std::cout << "Global tree sum: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "Global tree sum: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }
}
