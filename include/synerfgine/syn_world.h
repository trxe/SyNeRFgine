#pragma once

#include <synerfgine/bufferset.cuh>
#include <synerfgine/camera.h>
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

class SyntheticWorld {
public:
    SyntheticWorld();
    bool handle_user_input(const ivec2& resolution);
    bool handle(CudaDevice& device, const ivec2& resolution);
    bool shoot_network(CudaDevice& device, const ivec2& resolution, ngp::Testbed& testbed);
    bool debug_visualize_pos(CudaDevice& device, const vec3& pos, const vec3& col, float sphere_size = 1.0f);
    void create_object(const std::string& filename);
    void imgui(float frame_time);

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

private:
    friend class sng::Display;
	bool handle_user_input();
    // void draw_object(CudaDevice& device, VirtualObject& vo);
    // void shade_object(CudaDevice& device, VirtualObject& vo);
    void draw_object_async(CudaDevice& device, VirtualObject& vo);
    std::optional<VirtualObject> m_object;
    // std::unordered_map<std::string, VirtualObject> m_objects;

    int m_kernel_size{10};
    float m_std_dev{2.0f};
    ImgFilters m_filter_type{ImgFilters::None};
    bool m_show_kernel_settings{true};

    Camera m_camera;
    mat4x3 m_last_camera;
    Light m_sun;
    bool m_is_dirty;
    bool m_display_shadow{true};
    GPUMemory<NerfPayload> m_nerf_payloads;
    GPUMemory<float> m_shadow_coeffs;

    // Buffers and resolution
	std::shared_ptr<GLTexture> m_rgba_render_textures;
	std::shared_ptr<GLTexture> m_depth_render_textures;
	std::shared_ptr<CudaRenderBuffer> m_render_buffer;
	CudaRenderBufferView m_render_buffer_view;
    ivec2 m_resolution;
};

}