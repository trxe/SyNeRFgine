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

}