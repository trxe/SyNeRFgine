#pragma once
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_device.h>

namespace ngp {
namespace pt {

using namespace tcnn;

struct Camera {
    vec3 eye;
    vec3 at;
    vec3 up;
    float fov;
    ivec2 resolution;
    float aperture;
    float focus_dist;
};

}
}
