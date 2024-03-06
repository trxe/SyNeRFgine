#pragma once

#include <synerfgine/bufferset.cuh>
#include <synerfgine/camera.cuh>
#include <synerfgine/cuda_helpers.h>
#include <synerfgine/filters.cuh>
#include <synerfgine/light.cuh>
#include <synerfgine/virtual_object.h>

#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/render_buffer.h>
#include <neural-graphics-primitives/testbed.h>

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_host.h>
#include <tiny-cuda-nn/gpu_memory.h>

#include <optional>
#include <vector>

namespace sng {

using ngp::CudaRenderBuffer;
using ngp::GLTexture;
using tcnn::GPUMemory;

class Display;

enum ImgBuffers {
    Final,
    ReflPayloadOrigin,
    ReflPayloadDir,
    ReflPayloadT,
    WorldOrigin,
    WorldDir,
    WorldNormal,
    ReflRGBA,
    ReflDepth,
};

static const char* buffer_names[] = {
    "Final",
    "[Refl][Payload] Origin",
    "[Refl][Payload] Dir",
    "[Refl][Payload] t",
    "[World] Origin - Post",
    "[World] Dir - Post",
    "[World] Normal - Post",
    "[Refl] RGBA",
    "[Refl] Depth",
};

class SyntheticWorld {
public:
    SyntheticWorld();
    ~SyntheticWorld();
    bool handle_user_input(const ivec2& resolution);
    bool handle(CudaDevice& device, const ivec2& resolution);
    void reset_frame(CudaDevice& device, const ivec2& resolution);
    void generate_rays_async(CudaDevice& device);
    bool shoot_network(CudaDevice& device, const ivec2& resolution, ngp::Testbed& testbed);
    void debug_visualize_pos(CudaDevice& device, const vec3& pos, const vec3& col, float sphere_size = 1.0f);
    void create_object(const std::string& filename);
    void imgui(float frame_time);

    void debug_init_rays(CudaDevice& device, const ivec2& resolution);

    // inline std::unordered_map<std::string, VirtualObject>& objects() { return m_objects; }
    // inline void delete_object(const std::string& name) { m_objects.erase(name); }
    void delete_object() { m_object.reset(); }
    inline std::optional<VirtualObject>& get_object() { return m_object; }
    inline std::shared_ptr<CudaRenderBuffer> render_buffer() { return m_render_buffer; }
    inline const Camera& camera() { return m_camera; }
    inline Camera& mut_camera() { return m_camera; }

	inline vec3 sun_pos() const { return m_sun.pos; }
	inline Light sun() const { return m_sun; }
    inline Triangle* gpu_triangles() { 
        return m_object.has_value() ? m_object.value().gpu_triangles() : nullptr;
    }
    inline size_t gpu_triangles_count() { 
        return m_object.has_value() ? m_object.value().cpu_triangles().size() : 0;
    }

    inline void resize_gpu_buffers(uint32_t n_elements) {
        m_gpu_positions.check_guards();
        m_gpu_positions.resize(n_elements);
        m_gpu_directions.check_guards();
        m_gpu_directions.resize(n_elements);
        m_gpu_normals.check_guards();
        m_gpu_normals.resize(n_elements);
        m_gpu_scatters.check_guards();
        m_gpu_scatters.resize(n_elements);
        m_nerf_payloads.check_guards();
        m_nerf_payloads.resize(n_elements);
        m_nerf_payloads_refl.check_guards();
        m_nerf_payloads_refl.resize(n_elements);
        m_shadow_coeffs.check_guards();
        m_shadow_coeffs.resize(n_elements);
    }

private:
    friend class sng::Display;
    void release();
    // void draw_object(CudaDevice& device, VirtualObject& vo);
    // void shade_object(CudaDevice& device, VirtualObject& vo);
    void draw_object_async(CudaDevice& device, VirtualObject& vo);
    std::optional<VirtualObject> m_object;
    // std::unordered_map<std::string, VirtualObject> m_objects;

    int m_kernel_size{10};
    float m_std_dev{2.0f};
    ImgFilters m_filter_type{ImgFilters::Box};
    ImgBuffers m_buffer_type{ImgBuffers::Final};
    bool m_show_kernel_settings{true};

    Camera m_camera;
    mat4x3 m_last_camera;
    Light m_sun;
    bool m_is_dirty;
    bool m_display_shadow{true};
    bool m_display_nerf_payload_refl{true};
    GPUMemory<NerfPayload> m_nerf_payloads;
    GPUMemory<NerfPayload> m_nerf_payloads_refl;
    GPUMemory<float> m_shadow_coeffs;
    GPUMemory<vec3> m_gpu_positions;
    GPUMemory<vec3> m_gpu_directions;
    GPUMemory<vec3> m_gpu_normals;
    GPUMemory<vec3> m_gpu_scatters;

    // Buffers and resolution
	std::shared_ptr<GLTexture> m_rgba_render_textures;
	std::shared_ptr<GLTexture> m_depth_render_textures;
	std::shared_ptr<CudaRenderBuffer> m_render_buffer;
	CudaRenderBufferView m_render_buffer_view;
    ivec2 m_resolution;
    bool is_buffer_outdated{true};
};

}