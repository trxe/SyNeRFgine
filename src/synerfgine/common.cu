#include <synerfgine/common.cuh>

namespace sng {

__global__ void debug_shade(uint32_t n_elements, vec4* __restrict__ rgba, vec3 color, float* __restrict__ depth, float depth_value) {
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_elements) return;
    rgba[idx] = color;
    depth[idx] = depth_value;
}

__global__ void print_shade(uint32_t n_elements, vec4* __restrict__ rgba, float* __restrict__ depth) {
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_elements) return;
    if (idx % 100 == 0) {
        vec4 color = rgba[idx];
        printf("%d: %f, %f, %f, %f | %f\n", idx, color.r, color.g, color.b, color.a, depth[idx]);
    }
}

__global__ void init_rand_state(uint32_t n_elements, curandState_t* rand_state) {
    uint64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_elements) return;
    curand_init(static_cast<uint64_t>(PT_SEED),  idx, (uint64_t)0, rand_state+idx);
}

__device__ vec4 box_filter_vec4(uint32_t idx, ivec2 resolution, vec4* __restrict__ buffer, int kernel_size) {
    vec4 sum{};
    int nidx = idx;
    float z = 0.0;
    int x = idx % resolution.x, xmin = max(0, x-kernel_size), xmax = min(resolution.x-1, x+kernel_size);
    int y = idx / resolution.x, ymin = max(0, y-kernel_size), ymax = min(resolution.y-1, y+kernel_size);
    for (int i = xmin; i <= xmax; ++i) {
        for (int j = ymin; j <= ymax; ++j) {
            nidx = i * resolution.x + j;
            sum += buffer[nidx];
            z += 1.0f;
        }
    }
    return sum / z;
}

__global__ void debug_uv_shade(uint32_t n_elements, vec4* __restrict__ rgba, float* __restrict__ depth, ivec2 resolution) {
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_elements) return;
    float x = (float)(idx % resolution.x) / (float) resolution.x;
    float y = (float)(idx / resolution.x) / (float) resolution.y;
    rgba[idx] = {x, y, 0.0, 1.0};
    depth[idx] = 1.0;
}
}