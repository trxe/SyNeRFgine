#pragma once
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_device.h>
#include "common.cuh"

namespace ngp {
namespace pt {

struct Transform {
    vec3 pos;
    vec3 rot;
    float scale;
    mat3 rotate_mat;

};

}
}

