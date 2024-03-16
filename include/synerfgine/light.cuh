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
    uint32_t id;
    vec3 pos;
    vec3 color;
    float intensity;

};

}