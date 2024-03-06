#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_host.h>
#include <tiny-cuda-nn/vec.h>

namespace sng {

using namespace tcnn;

struct Material {
    vec3 ka;
    vec3 kd;
    vec3 ks;
    float n;
};
    
} // namespace sng

