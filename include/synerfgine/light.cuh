#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <synerfgine/common.cuh>
#include <fmt/format.h>

namespace sng {

enum LightType {
    Point,
    Directional,
};

struct Light {
    __host__ Light(uint32_t id, const nlohmann::json& config) : id(id), color(1.0), type(LightType::Point) {
        // std::string type_str {config["type"].get<std::string>()}; 
        auto& a = config["pos"];
        pos = { a[0].get<float>(), a[1].get<float>(), a[2].get<float>() };
        intensity  = config["intensity"].get<float>(); 
        size  = config["size"].get<float>(); 
        std::string type_str = "point";
        if (config.count("type")) {
            type_str = config["type"].get<std::string>();
            if (type_str == "point") type = LightType::Point;
            else if (type_str == "directional") type = LightType::Directional;
            else throw std::runtime_error(fmt::format("{} light not recognized", type_str.c_str()));
        }
        if (config.count("anim")) {
            anim_start_pos = pos;
            auto& a = config["anim"]["end"];
            anim_end_pos = { a[0].get<float>(), a[1].get<float>(), a[2].get<float>() };
            anim_step_size = config["anim"]["step"];
            anim_ratio = 0.0;
        }
        tlog::success() << "Set " << type_str << " light position " << pos << " at intensity " << intensity;
    }

    void next_frame(const float& speed) {
        if (anim_step_size == 0.0f) return;
        float next_ratio = anim_ratio + anim_step_size;
        if (next_ratio > 1.0 || next_ratio < 0.0) {
            anim_step_size = -anim_step_size;
            next_ratio = anim_ratio + anim_step_size;
        }
        anim_ratio = next_ratio;
        pos = (1.0f - anim_ratio) * anim_start_pos + anim_ratio * anim_end_pos;
    }

    __host__ void imgui() {
        // std::string unique_pos = fmt::format("[{}] pos", id);
        std::string unique_int = fmt::format("[{}] intensity", id);
        std::string unique_size = fmt::format("[{}] area size", id);
        std::string title = fmt::format("Light [{}]", id);
        std::string unique_val = fmt::format("LVals [{}]", id);
        std::string info = fmt::format("\"pos\" : [{:.3f}, {:.3f}, {:.3f}],", pos.x, pos.y, pos.z);
        if (ImGui::TreeNode(title.c_str())) {
            ImGui::InputTextMultiline(unique_val.c_str(), info.data(), info.size() + 1, {300, 20});
            if (ImGui::SliderFloat(unique_int.c_str(), &intensity, 0.0, 1.0)) { is_dirty = true; }
            if (ImGui::SliderFloat(unique_size.c_str(), &size, 0.0, 1.0)) { is_dirty = true; }
            ImGui::TreePop();
        }
        ImGui::Separator();
    }

    __device__ vec3 sample() const {
        return pos;
    }

    __device__ vec3 sample(curandState_t& rand_state, float multiplier = 1.0f) const {
        vec3 offset = {
            fractf(curand_uniform(&rand_state)),
            fractf(curand_uniform(&rand_state)),
            fractf(curand_uniform(&rand_state)) };
        return pos + offset * size * multiplier;
    }

    uint32_t id;
    vec3 pos;
    vec3 color;
    LightType type;
    float intensity;
    float size;
    bool is_dirty{true};

    vec3 anim_start_pos;
    vec3 anim_end_pos;
    float anim_ratio;
    float anim_step_size;
};

}