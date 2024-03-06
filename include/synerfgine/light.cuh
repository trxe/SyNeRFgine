#pragma once 


#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/random_val.cuh>

#include <synerfgine/input.h>

#include <tiny-cuda-nn/common.h>

namespace sng {

using namespace tcnn;

namespace light_default {
const uint32_t fov_axis = 1;
const vec3 up_dir{0.0f, 1.0f, 0.0f};
}

struct Light {
public:
	Light(): pos(3.0f), col(1.0f), intensity(1.0f) {}
    Light(const vec3& pos, const vec3& col, float intensity) : pos(pos), col(col), intensity(intensity) {}
    vec3 pos;
    vec3 col;
    float intensity;
    bool operator==(const Light& other) {
        return other.pos == pos && other.col == col && other.intensity && intensity && other.m_fov_axis == m_fov_axis;
    }
    bool handle_user_input(const ivec2& resolution) {
        vec2 rel = vec2{ImGui::GetIO().MouseDelta.x, ImGui::GetIO().MouseDelta.y} / (float)resolution[m_fov_axis];
        // Right held
        if (ImGui::GetIO().MouseDown[1]) {
            mat3 rot = rotation_from_angles(-rel * 2.0f * ngp::PI());
            pos = transpose(rot) * pos;
            return true;
        }
        return false;
    }
private:
    mat3 rotation_from_angles(const vec2& angles) const {
        vec3 up = light_default::up_dir;
        vec3 side = normalize(cross(pos, up));
        return rotmat(angles.x, up) * rotmat(angles.y, side);
    }
	uint32_t m_fov_axis = light_default::fov_axis;
};

}