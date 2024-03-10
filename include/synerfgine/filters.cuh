#pragma once
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_device.h>

using namespace tcnn;

__device__ float median_kernel(float* __restrict__ values, ivec2 resolution, int x, int y, int inner_radius, int outer_radius);
__device__ float gaussian_kernel(float* __restrict__ values, ivec2 resolution, int x, int y, int kernel_size, float SD);
__device__ float pull_push_kernel(float* __restrict__ values, ivec2 resolution, int x, int y);
__device__ vec4 pull_push_kernel_rgba(vec4* __restrict__ values, ivec2 resolution, int x, int y, int rounds);

namespace sng {

enum ImgFilters {
    None = 0,
    Box = 1,
    Gaussian = 2,
    PullPush = 3
};

static constexpr const char* filter_names[] = {
    "None",
    "Box/Median",
    "Gaussian",
    "PullPush",
};

}