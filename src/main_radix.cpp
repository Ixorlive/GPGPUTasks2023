#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/radix_cl.h"
#include "libgpu/work_size.h"

#include <iostream>
#include <stdexcept>
#include <vector>


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

constexpr uint NUM_BITS = 4;
constexpr uint TILE_SIZE = 16;

int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    unsigned int n = 32 * 1024 * 1024;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<int>::max());
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<unsigned int> cpu_sorted;
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cpu_sorted = as;
            std::sort(cpu_sorted.begin(), cpu_sorted.end());
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;
    }

    gpu::gpu_mem_32u as_gpu;
    as_gpu.resizeN(n);

    std::string defines = "-DTILE_SIZE=" + std::to_string(TILE_SIZE) + " -DNUM_BITS=" + std::to_string(NUM_BITS);
    {
        ocl::Kernel radix(radix_kernel, radix_kernel_length, "radix", defines);
        ocl::Kernel sort_workgroup(radix_kernel, radix_kernel_length, "sort_workgroup", defines);
        ocl::Kernel matrix_transpose(radix_kernel, radix_kernel_length, "matrix_transpose", defines);
        ocl::Kernel fill_count_matrix(radix_kernel, radix_kernel_length, "fill_count_matrix", defines);
        ocl::Kernel prefix_sum(radix_kernel, radix_kernel_length, "prefix_sum", defines);
        radix.compile();
        sort_workgroup.compile();
        matrix_transpose.compile();
        prefix_sum.compile();
        fill_count_matrix.compile();

        unsigned int workGroupSize = 128;
        unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        const gpu::WorkSize workSize(workGroupSize, global_work_size);

        gpu::gpu_mem_32u bs_gpu;
        bs_gpu.resizeN(n);

        const uint K = 1 << NUM_BITS;                          // range of values [0..K - 1]
        const uint M = (n + workGroupSize - 1) / workGroupSize;// number of work groups

        gpu::gpu_mem_32u matrix_gpu;
        gpu::gpu_mem_32u temp_matrix_gpu;
        matrix_gpu.resizeN(K * M);
        temp_matrix_gpu.resizeN(K * M);

        const gpu::WorkSize prefix_workSize(workGroupSize, (K * M + workGroupSize - 1) / workGroupSize * workGroupSize);
        const gpu::WorkSize matrix_workSize(TILE_SIZE, TILE_SIZE, K, M);

        constexpr uint numIters = sizeof(uint) * 8 / NUM_BITS;
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных
            for (uint i = 0; i < numIters; i++) {
                // stable sorting to each workgroup
                sort_workgroup.exec(workSize, as_gpu, bs_gpu, n, i);
                std::swap(as_gpu, bs_gpu);
                // filling out the matrix with counting each value in the array within the work group
                fill_count_matrix.exec(workSize, as_gpu, matrix_gpu, n, i);
                // transpose matrix
                matrix_transpose.exec(matrix_workSize, matrix_gpu, temp_matrix_gpu, M, K);
                std::swap(temp_matrix_gpu, matrix_gpu);
                // calculating prefix sum in matrix
                for (uint i = 1, d = 1; i < n; i <<= 1, ++d) {
                    prefix_sum.exec(prefix_workSize, matrix_gpu, temp_matrix_gpu, K * M, d);
                    std::swap(matrix_gpu, temp_matrix_gpu);
                }
                // sort (reordering)
                radix.exec(workSize, as_gpu, matrix_gpu, bs_gpu, n, K, M, i);
                std::swap(as_gpu, bs_gpu);
            }
            t.nextLap();
        }
        t.stop();
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;

        as_gpu.readN(as.data(), n);
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }
    return 0;
}
