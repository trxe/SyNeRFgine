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
        tlog::success() << "Set point light position " << pos << " at intensity " << intensity;
    }

    __host__ void imgui() {
        std::string unique_pos = fmt::format("[{}] pos", id);
        std::string unique_int = fmt::format("[{}] intensity", id);
        std::string title = fmt::format("Light [{}]", id);
        if (ImGui::TreeNode(title.c_str())) {
            if (ImGui::SliderFloat3(unique_pos.c_str(), pos.data(), -2.0, 2.0)) {
                is_dirty = true;
            }
            if (ImGui::SliderFloat(unique_int.c_str(), &intensity, 0.0, 1.0)) {
                is_dirty = true;
            }
            ImGui::TreePop();
        }
        ImGui::Separator();
    }

    uint32_t id;
    vec3 pos;
    vec3 color;
    float intensity;
    bool is_dirty{true};

};

}