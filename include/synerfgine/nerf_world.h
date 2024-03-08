#pragma once

#include <neural-graphics-primitives/testbed.h>
#include <synerfgine/cuda_helpers.h>
#include <synerfgine/camera.h>
#include <synerfgine/virtual_object.h>
#include <synerfgine/light.cuh>

namespace sng {
using ngp::Testbed;

class Display;

class NerfWorld {
public:
    NerfWorld();
    void init(Testbed* testbed);
    bool handle(CudaDevice& device, const Camera& cam, const Light& sun, std::optional<VirtualObject>& vo, const ivec2& view_res);
    void shadow_rays(
        cudaStream_t stream,
        const mat4x3& camera_matrix0,
        const mat4x3& camera_matrix1,
        const mat4x3& prev_camera_matrix,
        const vec2& orig_screen_center,
        const vec2& relative_focal_length,
        CudaRenderBuffer& render_buffer,
        CudaDevice* device
    );
    std::shared_ptr<CudaRenderBuffer> render_buffer() { return m_render_buffer; }
    void imgui(float frame_time);

private:
    friend class sng::Display;
    void shadow_rays(
        cudaStream_t stream,
        const mat4x3& camera_matrix0,
        const mat4x3& camera_matrix1,
        const mat4x3& prev_camera_matrix,
        const vec2& orig_screen_center,
        const vec2& relative_focal_length,
        CudaRenderBufferView& render_buffer,
        CudaDevice* device
    );
    mat4x3 m_last_camera;
    mat4x4 m_last_vo;
    Light m_last_sun;
    float m_render_ms{1.0};
    float m_dynamic_res_target_fps{25.0};
    int m_fixed_res_factor{36};
    bool m_display_shadow{true};
    bool m_is_dirty{true};

    Testbed *m_testbed;
    std::shared_ptr<GLTexture> m_rgba_render_textures;
    std::shared_ptr<GLTexture> m_depth_render_textures;
    std::shared_ptr<CudaRenderBuffer> m_render_buffer;
    CudaRenderBufferView m_render_buffer_view;
    ivec2 m_resolution;
};

}