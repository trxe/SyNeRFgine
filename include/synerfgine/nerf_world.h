#pragma once


#include <synerfgine/camera.h>
#include <synerfgine/cuda_helpers.h>

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/nerf.h>
#include <neural-graphics-primitives/nerf_loader.h>
#include <neural-graphics-primitives/nerf_network.h>
#include <neural-graphics-primitives/random_val.cuh>
#include <neural-graphics-primitives/render_buffer.h>
#include <neural-graphics-primitives/testbed.h>

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_host.h>
#include <tiny-cuda-nn/gpu_memory.h>

#include <json/json.hpp>

#include <filesystem/path.h>

#include <memory>

namespace sng {

using namespace tcnn;
namespace fs = filesystem;
using NerfNet = ngp::NerfNetwork<tcnn::network_precision_t>;
using json = nlohmann::json;

static const size_t SNAPSHOT_FORMAT_VERSION = 1;
void update_density_grid_mean_and_bitfield(cudaStream_t stream, Testbed::Nerf& nerf);

class NerfWorld {
public:
    // NerfWorld();
    bool handle(CudaDevice& device, const ivec2& resolution);
    void load_snapshot(const nlohmann::json& config, const fs::path& data_path);
    // void load_training_data(const nlohmann::json& config);
    const Camera& camera() { return m_camera; }
    Camera& mut_camera() { return m_camera; }

private:
    ngp::Testbed::NerfTracer m_tracer{};
    std::shared_ptr<NerfNet> m_nerf_network{};
    fs::path m_data_path;
    json m_network_info{};
    ivec2 m_resolution{};
    Camera m_camera{};
	ngp::BoundingBox m_aabb;
	ngp::BoundingBox m_raw_aabb;
	ngp::BoundingBox m_render_aabb;
    mat3 m_render_aabb_to_local;
    ivec2 m_screen_center{};
	float m_bounding_radius = 1.0f;
    uint32_t seed = 1337;
    ngp::default_rng_t m_rng{seed};
    vec3 m_up_dir;
	ngp::Testbed::Nerf m_nerf;

};

}