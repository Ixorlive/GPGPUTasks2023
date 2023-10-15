#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>
#include <vector>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/prefix_sum_cl.h"
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
    unsigned int max_n = (1 << 24);

    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    for (unsigned int n = 4096; n <= max_n; n *= 4) {
        std::cout << "______________________________________________" << std::endl;
        unsigned int values_range = std::min<unsigned int>(1023, std::numeric_limits<int>::max() / n);
        std::cout << "n=" << n << " values in range: [" << 0 << "; " << values_range << "]" << std::endl;

        std::vector<unsigned int> as(n, 0);
        FastRandom r(n);
        for (int i = 0; i < n; ++i) {
            as[i] = r.next(0, values_range);
        }

        std::vector<unsigned int> bs(n, 0);
        {
            for (int i = 0; i < n; ++i) {
                bs[i] = as[i];
                if (i) {
                    bs[i] += bs[i - 1];
                }
            }
        }
        const std::vector<unsigned int> reference_result = bs;

        {
            {
                std::vector<unsigned int> result(n);
                for (int i = 0; i < n; ++i) {
                    result[i] = as[i];
                    if (i) {
                        result[i] += result[i - 1];
                    }
                }
                for (int i = 0; i < n; ++i) {
                    EXPECT_THE_SAME(reference_result[i], result[i], "CPU result should be consistent!");
                }
            }

            std::vector<unsigned int> result(n);
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                for (int i = 0; i < n; ++i) {
                    result[i] = as[i];
                    if (i) {
                        result[i] += result[i - 1];
                    }
                }
                t.nextLap();
            }
            std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "CPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            ocl::Kernel prefix(prefix_sum_kernel, prefix_sum_kernel_length, "prefix_sum_naive");
            gpu::gpu_mem_32u as_gpu;
            gpu::gpu_mem_32u res_gpu;
            as_gpu.resizeN(n);
            res_gpu.resizeN(n);
            prefix.compile();
            unsigned int workGroupSize = 128;
            unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
            gpu::WorkSize work_size(workGroupSize, global_work_size);
            timer t;
            for (size_t i = 0; i < benchmarkingIters; i++) {
                as_gpu.writeN(as.data(), n);
                t.restart();
                for (uint i = 1, d = 1; i < n; i <<= 1, ++d) {
                    prefix.exec(work_size, as_gpu, res_gpu, n, d);
                    std::swap(as_gpu, res_gpu);
                }
                t.nextLap();
            }
            t.stop();
            std::cout << "GPU (naive scan): " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU (naive scan): " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
            std::vector<uint> result(n);
            as_gpu.readN(result.data(), n);

            for (int i = 0; i < n; ++i) {
                EXPECT_THE_SAME(reference_result[i], result[i], "GPU result should be consistent!");
            }
        }
        {
            ocl::Kernel reduce(prefix_sum_kernel, prefix_sum_kernel_length, "reduce");
            ocl::Kernel downsweep(prefix_sum_kernel, prefix_sum_kernel_length, "downsweep");
            gpu::gpu_mem_32u as_gpu;
            gpu::gpu_mem_32u last_el_gpu;
            as_gpu.resizeN(n);
            last_el_gpu.resizeN(1);
            reduce.compile();
            downsweep.compile();

            unsigned int workGroupSize = 128;
            unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
            gpu::WorkSize work_size(workGroupSize, global_work_size);
            timer t;

            unsigned int last_element = 0;
            for (size_t i = 0; i < benchmarkingIters; i++) {
                as_gpu.writeN(as.data(), n);
                t.restart();
                for (uint d = 1; d < n; d <<= 1) {
                    reduce.exec(work_size, as_gpu, last_el_gpu, n, d);
                }
                last_el_gpu.readN(&last_element, 1);
                for (uint d = n >> 1; d > 0; d >>= 1) {
                    downsweep.exec(work_size, as_gpu, n, d);
                }
                t.nextLap();
            }
            t.stop();
            std::cout << "GPU (efficient scan): " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU (efficient scan):" << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
            std::vector<uint> result(n + 1);
            as_gpu.readN(result.data(), n);
            result.back() = last_element;
            for (int i = 0; i < n; ++i) {
                EXPECT_THE_SAME(reference_result[i], result[i + 1], "GPU result should be consistent!");
            }
        }
    }
}
