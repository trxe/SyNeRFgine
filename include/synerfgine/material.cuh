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

namespace sng {
using namespace tcnn;
using ngp::Ray;

enum MaterialType {
    Lambertian,
    Glossy,
};

struct Material {
    NGP_HOST_DEVICE Material(uint32_t id, const vec3& kd, float rg, float n, MaterialType type) : 
        id(id), kd(kd), ka(kd * 0.01f), ks(1.0), rg(rg), n(n), type(type) {}

    __host__ Material(uint32_t id, const nlohmann::json& config) : id(id), ks(1.0) {
        std::string type_str {config["type"].get<std::string>()}; 
        {
            auto& a = config["kd"];
            kd = { a[0].get<float>(), a[1].get<float>(), a[2].get<float>() };
        }
        if (config.count("ka")) {
            auto& a = config["ka"];
            ka = { a[0].get<float>(), a[1].get<float>(), a[2].get<float>() };
        }
        if (config.count("ks")) {
            auto& a = config["ks"];
            ks = { a[0].get<float>(), a[1].get<float>(), a[2].get<float>() };
        }
        n  = config["n"].get<float>(); 
        rg = config.count("rg") ? config["rg"].get<float>() : 0.0;
        if (type_str == "lambertian") {
            type = MaterialType::Lambertian;
            spec_angle = 0.0f;
        } else if (type_str == "glossy") {
            type = MaterialType::Glossy;
            spec_angle = config.count("spec_angle") ? config["spec_angle"].get<float>() : 0.001;
        } else {
            throw std::runtime_error(fmt::format("Material type {} not supported", type_str));
        }
        tlog::success() << "Set [" << type_str << "] material " << id << " ka = " << ka << ";  kd = "<<  kd << ";  n = " << n << "; ";
    }

    __host__ void imgui() {
        std::string unique_kd = fmt::format("[{}] kd", id);
        std::string unique_rg = fmt::format("[{}] rg", id);
        std::string unique_n = fmt::format("[{}] n", id);
        std::string unique_spec_angle = fmt::format("[{}] spec_angle", id);
        std::string title = fmt::format("Material [{}]", id);
        if (ImGui::TreeNode(title.c_str())) {
            if (ImGui::ColorPicker3(unique_kd.c_str(), kd.data())) {
                ka = kd;
                ka *= 0.00f;
                is_dirty = true;
            }
            if (ImGui::SliderFloat(unique_n.c_str(), &n, 0.0, 256.0)) {
                is_dirty = true;
            }
            if (ImGui::SliderFloat(unique_rg.c_str(), &rg, 0.0, 1.0)) {
                is_dirty = true;
            }
            if (ImGui::SliderFloat(unique_spec_angle.c_str(), &spec_angle, 0.0, 1.0)) {
                is_dirty = true;
            }
            ImGui::TreePop();
        }
        ImGui::Separator();
    }

    NGP_HOST_DEVICE void print() const {
        char ctype = type == MaterialType::Lambertian ? 'L' : '?';
        printf("#%d [%c] ka: {%f, %f, %f}, kd: {%f, %f, %f}, ks: {%f, %f, %f}, rg: %f, n: %f\n", 
            id,
            ctype,
            ka[0], ka[1], ka[2], 
            kd[0], kd[1], kd[2], 
            ks[0], ks[1], ks[2],
            rg,
            n
        );
    }

    __device__ vec3 local_color(const vec3& L, const vec3& N, const vec3& R, const vec3& V, const Light& light) const {
        return max(0.0f, dot(L, N)) * kd * light.intensity + pow(max(0.0f, dot(R, V)), n) * ks;
    }

    __device__ bool scatter(const HitRecord& hit_info, const vec3& src_dir, SampledRay& ray) const {
        ray.pos = hit_info.pos;
        if (type == Lambertian) {
            ray.dir = reflect(src_dir, hit_info.normal);
            ray.pdf = 1.0f / tcnn::PI;
            ray.attenuation *= rg;
        } else {
            return false;
        }
        return true;
    }

    __device__ bool scatter(const HitRecord& hit_info, const vec3& src_dir, SampledRay& ray, curandState& rand) const {
        ray.pos = hit_info.pos;
        if (type == Lambertian) {
            ray.dir = Rand::random_unit_vector(&rand);
            if (dot(ray.dir, hit_info.normal) < 0.0) {
                ray.dir = -ray.dir;
            }
            ray.pdf = 1.0f / tcnn::PI;
            ray.attenuation *= rg;
        } else if (type == Glossy) {
            ray.dir = reflect(-src_dir, hit_info.normal);
            auto longi = curand_uniform(&rand) * spec_angle;
            auto latid = curand_uniform(&rand) * 2.0 * tcnn::PI;
            ray.dir = sng::cone_random(ray.dir, hit_info.normal, longi, latid);
            ray.pdf = 1.0f / max(1.0f, spec_angle);
            ray.attenuation *= rg;
        } else {
            return false;
        }
        return true;
    }

    uint32_t id;
    vec3 ka{0.0};
    vec3 kd{0.0};
    vec3 ks{1.0};
    float n{1};
    float rg{0.9};
    MaterialType type;
    float spec_angle{0.001};
    bool is_dirty{true};
};

}