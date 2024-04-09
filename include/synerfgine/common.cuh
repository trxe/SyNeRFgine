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
#define MAX_SHADOW_SAMPLES 32

namespace sng {

using namespace ngp;
namespace fs = filesystem;

__host__ __device__ inline float ffmin(float a, float b) { return a < b ? a : b; }
__host__ __device__ inline float ffmax(float a, float b) { return a > b ? a : b; }
__host__ __device__ inline vec3 reflect(const vec3& incident, const vec3& normal) { 
    return 2.0f * dot(incident, normal) * normal - incident;
}
__host__ __device__ inline vec3 cone_random(const vec3& orig, const mat3& perturb_frame, float longi, float latid) {
    vec3 offset = {cos(longi) * sin(latid), sin(longi) * sin(latid), cos(longi)};
    return orig + perturb_frame * offset;
}
__host__ __device__ inline vec3 cone_random(const vec3& orig, const vec3& up, float longi, float latid) {
    const vec3& N = normalize(orig);
    vec3 B = normalize(cross(N, up));
    vec3 T = cross(B, N);
    mat3 perturb_frame = {T, B, N};
    vec3 offset = {
        sin(longi) * cos(latid), 
        sin(longi) * sin(latid), 
        cos(longi)
    };
    return orig + perturb_frame * offset;
}
__host__ __device__ inline mat3 get_perturb_matrix(const vec3& tangent, const vec3& normal) {
    const vec3 T = normalize(tangent);
    const vec3 N = normalize(normal);
    const vec3 B = normalize(cross(N, T));
    return {T, B, N};
}
__host__ __device__ inline vec3 get_normal(const vec3& tangent, const vec3& binormal) {
    return normalize(cross(normalize(tangent), binormal));
}
__host__ __device__ inline ivec2 downscale_resolution(const ivec2& resolution, float scale) {
    return clamp(ivec2(vec2(resolution) * scale), resolution / 16, resolution);
}
__host__ __device__ inline ivec2 scale_resolution(const ivec2& resolution, float scale) {
    return clamp(ivec2(vec2(resolution) * scale), resolution / 16, resolution * 16);
}
__host__ __device__ inline int get_idx(ivec2 coord, const ivec2& res) {
    return coord.x + coord.y * res.x;
}

// From render_buffer.cu
__global__ void init_rand_state(uint32_t n_elements, curandState_t* rand_state);
__device__ vec4 box_filter_vec4(uint32_t idx, ivec2 resolution, vec4* __restrict__ buffer, int kernel_size);
__device__ vec3 sng_tonemap(vec3 x, ETonemapCurve curve);
__device__ vec3 random_unit_vector(curandState_t* rand_state);

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
__global__ void transform_payload(
	uint32_t n_elements,
	const vec3* __restrict__ src_origin,
	const vec3* __restrict__ src_dir,
	const vec3* __restrict__ src_normal,
	vec3* __restrict__ dst_origin,
	vec3* __restrict__ dst_dir,
	vec3* __restrict__ dst_normal,
	mat3 rotation,
	vec3 translation,
	float scale_val,
	bool o2w
);
__global__ void transfer_color(
	uint32_t n_elements,
	const vec3* __restrict__ src_color,
	const float* __restrict__ src_depth,
	vec4* __restrict__ frame_buffer,
	float* __restrict__ depth_buffer
);

enum WorldObjectType {
    None,
    LightObj,
    VirtualObjectObj
};

static const char * world_object_names[] = {
    "None", "Light", "Virtual Object"
};

struct HitRecord {
  public:
    vec3 pos{0.0f};
    vec3 normal{0.0f};
    mat3 perturb_matrix{mat3::identity()};
    float t{MAX_DEPTH()};
    int32_t material_idx{-1};
    int32_t object_idx{-1};
    bool front_face{true};
};

struct SampledRay {
    vec3 pos{0.0f};
    vec3 dir{0.0f};
    float pdf{0.0f};
    float attenuation{1.0f};
};

struct ObjectTransform {
	NGP_HOST_DEVICE ObjectTransform(TriangleBvhNode* g_node, Triangle* g_tris, const mat3& rot, const vec3& pos, const float& scale, const int32_t& mat_id) :
		g_node(g_node), g_tris(g_tris), rot(rot), pos(pos), scale(scale), mat_id(mat_id) {}
	TriangleBvhNode* g_node;
	Triangle* g_tris;
	mat3 rot;
	vec3 pos;
	float scale;
    int32_t mat_id;
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

__device__ float depth_test_world(const vec3& origin, const vec3& dir, const ObjectTransform* __restrict__ objects, const size_t& object_count, int32_t& out_obj_id);
__device__ float depth_test_world(const vec3& origin, const vec3& dir, const ObjectTransform* __restrict__ objects, const size_t& object_count, int32_t& out_obj_id, HitRecord& hit_info);
__device__ float depth_test_nerf(const float& full_d, const uint32_t& n_steps, const float& cone_angle_constant, const vec3& src, const vec3& L, const vec3& invL,
	const uint8_t* __restrict__ density_grid, const uint32_t& min_mip, const uint32_t& max_mip, const BoundingBox& render_aabb, const mat3& render_aabb_to_local);
__device__ float depth_test_nerf(const uint32_t& n_steps, const float& cone_angle_constant, const vec3& src, const vec3& dst,
	const uint8_t* __restrict__ density_grid, const uint32_t& min_mip, const uint32_t& max_mip, const BoundingBox& render_aabb, const mat3& render_aabb_to_local);
//  returns has_hit, depth and thickness
__device__ std::tuple<bool, float, float> depth_test_nerf_far(const uint32_t& n_steps, const float& cone_angle_constant, const vec3& src, const vec3& dst,
	const uint8_t* __restrict__ density_grid, const uint32_t& min_mip, const uint32_t& max_mip, const BoundingBox& render_aabb, const mat3& render_aabb_to_local);
__device__ std::pair<bool, float> depth_test_nerf_furthest(const uint32_t& n_steps, const float& cone_angle_constant, const vec3& src, const vec3& dst,
	const uint8_t* __restrict__ density_grid, const uint32_t& min_mip, const uint32_t& max_mip, const BoundingBox& render_aabb, const mat3& render_aabb_to_local, const float& max_depth);


}