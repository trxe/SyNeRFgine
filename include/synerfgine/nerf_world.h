#pragma once

#include <synerfgine/camera.h>
#include <synerfgine/cuda_helpers.h>

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/gpu_memory.h>
#include <neural-graphics-primitives/nerf_network.h>
#include <neural-graphics-primitives/render_buffer.h>
#include <neural-graphics-primitives/testbed.h>

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_host.h>

#include <json/json.hpp>

#include <filesystem>
#include <memory>

namespace sng {

using namespace tcnn;
namespace fs = std::filesystem;
using NerfNet = ngp::NerfNetwork<tcnn::network_precision_t>;
using json = nlohmann::json;

static const size_t SNAPSHOT_FORMAT_VERSION = 1;

class NerfWorld {
public:
    // NerfWorld();
    bool handle(CudaDevice& device, const ivec2& resolution);
    void load_network(const fs::path& path);
    const Camera& camera() { return m_camera; }
    Camera& mut_camera() { return m_camera; }

private:
    void load_snapshot(nlohmann::json config);

    ngp::Testbed::NerfTracer m_tracer{};
    std::shared_ptr<NerfNet> m_nerf_network{};
    json m_network_info{};
    ivec2 m_resolution{};
    Camera m_camera{};
};

}