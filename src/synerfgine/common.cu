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
        printf("%d: %f, %f, %f, %f | %f\n", idx, color.r, color.g, color.b, color.a, depth);
    }
}

__global__ void init_rand_state(uint32_t n_elements, curandState_t* rand_state) {
    uint64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_elements) return;
    curand_init(static_cast<uint64_t>(PT_SEED),  idx, (uint64_t)0, rand_state+idx);
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