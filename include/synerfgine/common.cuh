#pragma once
#include <curand_kernel.h>
#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/triangle.cuh>
#include <neural-graphics-primitives/triangle_bvh.cuh>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <deque>
#include <unordered_map>
#include <tinylogger/tinylogger.h>
#include <filesystem/path.h>
#include <fmt/format.h>
#include <json/json.hpp>

#define PT_SEED 1999

namespace sng {

using namespace ngp;
namespace fs = filesystem;

__host__ __device__ inline float ffmin(float a, float b) { return a < b ? a : b; }
__host__ __device__ inline float ffmax(float a, float b) { return a > b ? a : b; }
__host__ __device__ inline vec3 reflect(const vec3& incident, const vec3& normal) { 
    return 2.0f * dot(incident, normal) * normal - incident;
}
__host__ __device__ inline ivec2 downscale_resolution(const ivec2& resolution, float scale) {
    return clamp(ivec2(vec2(resolution) * scale), resolution / 16, resolution);
}
__host__ __device__ inline ivec2 scale_resolution(const ivec2& resolution, float scale) {
    return clamp(ivec2(vec2(resolution) * scale), resolution / 16, resolution * 16);
}

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
__global__ void debug_shade(uint32_t n_elements, vec4* __restrict__ rgba, vec3 color, float* __restrict__ depth, float depth_value);
__global__ void print_shade(uint32_t n_elements, vec4* __restrict__ rgba, float* __restrict__ depth);
__global__ void debug_uv_shade(uint32_t n_elements, vec4* __restrict__ rgba, float* __restrict__ depth, ivec2 resolution);

enum WorldObjectType {
    None,
    LightObj,
    VirtualObjectObj
};

static const char * world_object_names[] = {
    "None", "Light", "Virtual Object"
};

struct ObjectTransform {
	NGP_HOST_DEVICE ObjectTransform(TriangleBvhNode* g_node, Triangle* g_tris, const mat3& rot, const vec3& pos, const float& scale) :
		g_node(g_node), g_tris(g_tris), rot(rot), pos(pos), scale(scale) {}
	TriangleBvhNode* g_node;
	Triangle* g_tris;
	mat3 rot;
	vec3 pos;
	float scale;
};

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

  double get_ave_time(const char* label) {
    return rolling_averages[label];
  }
  double log_time(const char* label, bool is_print = false) {
    const auto t = get_time_ms();
    if (is_print)
        tlog::info() << "[" << label << "]: " << t.count() << "ms";
    records[label].push_back(t.count());
    double to_remove = records[label].front();
    if (records.size() > window) {
        records[label].pop_front();
    } else {
        to_remove = 0.0;
    }
    rolling_averages[label] = (rolling_averages[label] * ((float)records.size() - 1) - to_remove + t.count()) / (float)(records.size());
    return rolling_averages[label];
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
  std::unordered_map<const char*, std::deque<double>> records;
  std::unordered_map<const char*, double> rolling_averages;
  const uint32_t window = 100;
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

class File {
public:
    static std::string read_text(const std::string& fp) {
        if (!fs::path(fp).exists()) {
            throw std::runtime_error("Text File not found: " + fp);
        }

        std::ifstream file(fp);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + fp);
        }

        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        file.close();

        return content;
    }

    static nlohmann::json read_json(const std::string& fp) {
        if (!fs::path(fp).exists()) {
            throw std::runtime_error("JSON File not found: " + fp);
        }

        // Open the file for reading
        std::ifstream file(fp);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + fp);
        }
        return nlohmann::json::parse(file);
    }

    static inline fs::path get_root_dir() {
        fs::path root_dir = fs::path::getcwd();
        fs::path exists_in_root_dir = "scripts";
        for (const auto& candidate : {
            fs::path{"."}/exists_in_root_dir,
            fs::path{".."}/exists_in_root_dir,
            root_dir/exists_in_root_dir,
            root_dir/".."/exists_in_root_dir,
        }) {
            if (candidate.exists()) {
                return candidate.str();
            }
        }
        return {};
    }

};

static tlog::Stream& operator<<(tlog::Stream& ostr, const vec3& v) {
    return ostr << "[" << v.r << ", " << v.g << ", " << v.b << "]";
}


}