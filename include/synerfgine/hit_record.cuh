#pragma once
#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/common_device.cuh>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_device.h>

namespace sng {

struct HitRecord {
  public:
    vec3 pos;
    vec3 normal;
    float t;
    uint32_t material_idx;
    bool front_face;

    NGP_HOST_DEVICE void set_face_normal(const Ray& r, const vec3& outward_normal) {
        front_face = dot(r.d, outward_normal) < 0.0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

}