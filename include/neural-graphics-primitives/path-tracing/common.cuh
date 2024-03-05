#pragma once
#include <curand_kernel.h>
#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/common_device.cuh>
#define PT_SEED 1999
#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <tinylogger/tinylogger.h>

namespace ngp { 
namespace pt {

__host__ __device__ inline float ffmin(float a, float b) { return a < b ? a : b; }
__host__ __device__ inline float ffmax(float a, float b) { return a > b ? a : b; }

// Always initialize the benchmarker, __timer should not be redeclared in the
// current context.
#define INIT_BENCHMARK() Timer __timer;

// Benchmark fn (in ms) without any label
#define BENCHMARK(name, fn) \
  __timer.reset();    \
  fn;                 \
  __timer.log_time(name, false);

constexpr float PT_EPSILON = std::numeric_limits<float>::epsilon();
constexpr float PT_MIN_FLOAT = std::numeric_limits<float>::min();
constexpr float PT_MAX_FLOAT = std::numeric_limits<float>::max();
constexpr float PT_MAX_T = 16480.0f;

__device__ vec3 vec3_to_col(const vec3& v);
__device__ void print_vec(const vec3& v);
__device__ void print_mat4x3(const mat4x3& m);
__global__ void rand_init_pixels(int resx, int resy, curandState *rand_state);

class Timer {
 public:
  std::chrono::system_clock::time_point reset() {
    start = std::chrono::system_clock::now();
    return start;
  }
  std::chrono::duration<double> get_time() {
    const auto end = std::chrono::system_clock::now();
    return std::chrono::duration<double>{end - start};
  }
  std::chrono::duration<double, std::milli> get_time_ms() {
    const auto end = std::chrono::system_clock::now();
    return std::chrono::duration<double, std::milli>(end - start);
  }

  void log_time(const char* label, bool is_print = false) {
    const auto t = get_time_ms();
    if (is_print)
        tlog::info() << "[" << label << "]: " << t.count() << "ms";
    records[label].push_back(t.count());
  }

  ~Timer() {
    for (auto& record : records) {
        auto& rlist = record.second;
        double total_time = std::accumulate(rlist.begin(), rlist.end(), 0.0);
        double ave_time = total_time / rlist.size();
        tlog::info() << "AVE [" << record.first << "]: " << ave_time << "ms";
    }
  }

 private:
  std::chrono::system_clock::time_point start;
  std::unordered_map<const char*, std::vector<double>> records;
};


class Rand {
public:
    __host__ static inline vec3 random_unit_vector() {
        float x = (float)std::rand() / (float)RAND_MAX;
        float y = (float)std::rand() / (float)RAND_MAX;
        float z = (float)std::rand() / (float)RAND_MAX;
        vec3 rnd_vec = {x, y, z};
        return normalize(rnd_vec);
    }

    __device__ static inline vec3 random_unit_vector(curandState* random_seed) {
        vec3 tmp {
            curand_normal(random_seed),
            curand_normal(random_seed),
            curand_normal(random_seed)
        };
        return normalize(tmp);
    }

};

}
}