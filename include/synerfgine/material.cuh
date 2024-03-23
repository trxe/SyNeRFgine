#pragma once

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/common_device.cuh>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_device.h>
#include <json/json.hpp>
#include <fmt/format.h>

#include <imgui/imgui.h>
#include <imguizmo/ImGuizmo.h>

#include <stdio.h>
#include <synerfgine/common.cuh>
#include <synerfgine/hit_record.cuh>

namespace sng {
using namespace tcnn;
using ngp::Ray;

enum MaterialType {
    Lambertian
};

struct Material {
    NGP_HOST_DEVICE Material(uint32_t id, const vec3& kd, float n, MaterialType type) : 
        id(id), ka(kd), kd(kd), ks(1.0), n(n), type(type) {}

    __host__ Material(uint32_t id, const nlohmann::json& config) : id(id), ks(1.0) {
        std::string type_str {config["type"].get<std::string>()}; 
        auto& a = config["kd"];
        ka = kd = { a[0].get<float>(), a[1].get<float>(), a[2].get<float>() };
        ka *= 0.05f;
        n  = config["n"].get<float>(); 
        if (type_str == "lambertian") {
            type = MaterialType::Lambertian;
        } else {
            throw std::runtime_error(fmt::format("Material type {} not supported", type_str));
        }
        tlog::success() << "Set [" << type_str << "] material " << id << " ka = " << ka << ";  kd = "<<  kd << ";  n = " << n << "; ";
    }

    __host__ void imgui() {
        std::string unique_kd = fmt::format("[{}] kd", id);
        std::string unique_n = fmt::format("[{}] n", id);
        std::string title = fmt::format("Material [{}]", id);
        if (ImGui::TreeNode(title.c_str())) {
            if (ImGui::ColorPicker3(unique_kd.c_str(), kd.data())) {
                ka = kd;
                ka *= 0.05f;
                is_dirty = true;
            }
            if (ImGui::SliderFloat(unique_n.c_str(), &n, 0.0, 256.0)) {
                is_dirty = true;
            }
            ImGui::TreePop();
        }
        ImGui::Separator();
    }

    NGP_HOST_DEVICE void print() const {
        char ctype = type == MaterialType::Lambertian ? 'L' : '?';
        printf("#%d [%c] ka: {%f, %f, %f}, kd: {%f, %f, %f}, ks: {%f, %f, %f}, n: %f\n", 
            id,
            ctype,
            ka[0], ka[1], ka[2], 
            kd[0], kd[1], kd[2], 
            ks[0], ks[1], ks[2],
            n
        );
    }

    __device__ bool scatter(const vec3& ro, const vec3& normal, vec3& next_dir, curandState& rand) const {
        next_dir = normal + Rand::random_unit_vector(&rand);
        return true;
    }

    uint32_t id;
    vec3 ka;
    vec3 kd;
    vec3 ks;
    float n;
    MaterialType type;
    bool is_dirty{true};
};

}