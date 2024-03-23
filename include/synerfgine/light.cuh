#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <synerfgine/common.cuh>
#include <fmt/format.h>

namespace sng {

struct Light {
    __host__ Light(uint32_t id, const nlohmann::json& config) : id(id), color(1.0) {
        // std::string type_str {config["type"].get<std::string>()}; 
        auto& a = config["pos"];
        pos = { a[0].get<float>(), a[1].get<float>(), a[2].get<float>() };
        intensity  = config["intensity"].get<float>(); 
        size  = config["size"].get<float>(); 
        tlog::success() << "Set point light position " << pos << " at intensity " << intensity;
    }

    __host__ void imgui() {
        // std::string unique_pos = fmt::format("[{}] pos", id);
        std::string unique_int = fmt::format("[{}] intensity", id);
        std::string unique_size = fmt::format("[{}] area size", id);
        std::string title = fmt::format("Light [{}]", id);
        std::string unique_val = fmt::format("LVals [{}]", id);
        std::string info = fmt::format("pos : [{:.3f}, {:.3f}, {:.3f}],", pos.x, pos.y, pos.z);
        if (ImGui::TreeNode(title.c_str())) {
            ImGui::InputTextMultiline(unique_val.c_str(), info.data(), info.size() + 1, {300, 20});
            if (ImGui::SliderFloat(unique_int.c_str(), &intensity, 0.0, 1.0)) { is_dirty = true; }
            if (ImGui::SliderFloat(unique_size.c_str(), &size, 0.0, 1.0)) { is_dirty = true; }
            ImGui::TreePop();
        }
        ImGui::Separator();
    }

    __device__ vec3 sample(curandState_t& rand_state) const {
        vec3 offset = {
            fractf(curand_uniform(&rand_state)),
            fractf(curand_uniform(&rand_state)),
            fractf(curand_uniform(&rand_state)) };
        return pos + offset * size;
    }

    uint32_t id;
    vec3 pos;
    vec3 color;
    float intensity;
    float size;
    bool is_dirty{true};
};

}