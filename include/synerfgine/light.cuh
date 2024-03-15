#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <synerfgine/common.cuh>

namespace sng {

struct Light {
    vec3 pos;
    vec3 color;
    float intensity;
};

}