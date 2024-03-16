#pragma once

// #include <synerfgine/display.h>

#include <neural-graphics-primitives/render_buffer.h>
#include <neural-graphics-primitives/testbed.h>

#include <synerfgine/display.cuh>
#include <synerfgine/light.cuh>
#include <synerfgine/material.cuh>
#include <synerfgine/raytracer.cuh>
#include <synerfgine/virtual_object.cuh>

#include <string>
#include <vector>

namespace sng {

using ngp::Testbed;

class Engine {
public:
    void init(int res_width, int res_height, const std::string& frag_fp, Testbed* nerf);
    void set_virtual_world(const std::string& config_fp);
    bool frame();
    void redraw_next_frame() { m_is_dirty = true; }

private:
    void imgui();
    void imguizmo();
    void init_buffers();
    void try_resize();
    void set_dead() { m_display.set_dead(); }
    void sync(cudaStream_t stream) { 
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream)); 
    }
    Testbed::View& nerf_render_buffer_view() {
        auto& view = m_testbed->m_views.front();
        view.device = &(m_testbed->primary_device());
        view.full_resolution = m_testbed->m_window_res;
        view.camera0 = m_testbed->m_smoothed_camera;
        // Motion blur over the fraction of time that the shutter is open. Interpolate in log-space to preserve rotations.
        view.camera1 = m_testbed->m_camera_path.rendering ? camera_log_lerp(m_testbed->m_smoothed_camera, m_testbed->m_camera_path.render_frame_end_camera, m_testbed->m_camera_path.render_settings.shutter_fraction) : view.camera0;
        view.visualized_dimension = m_testbed->m_visualized_dimension;
        view.relative_focal_length = m_testbed->m_relative_focal_length;
        view.screen_center = m_testbed->m_screen_center;
        view.render_buffer->set_hidden_area_mask(nullptr);
        view.foveation = {};
        view.render_aabb = m_testbed->m_render_aabb;
        view.render_aabb_to_local = m_testbed->m_render_aabb_to_local;
        view.min_mip = 0;
        view.max_mip = m_testbed->m_nerf.max_cascade;
        view.cone_angle_constant = m_testbed->m_nerf.cone_angle_constant;
        m_testbed->m_nerf.density_grid.data();
        return view;
    }

    bool m_is_dirty = true;
    Testbed* m_testbed;
    Display m_display;
    ivec2 m_next_frame_resolution;

    RayTracer m_raytracer;
    std::vector<Material> m_materials;
    std::vector<VirtualObject> m_objects;
    std::vector<Light> m_lights;

    std::vector<vec4> m_nerf_rgba_cpu;
    std::vector<float> m_nerf_depth_cpu;
    std::vector<vec4> m_syn_rgba_cpu;
    std::vector<float> m_syn_depth_cpu;

    INIT_BENCHMARK();
	float m_render_ms{30.0f};
	float m_last_target_fps{0.0f};

    // for imguizmo
    WorldObjectType m_transform_type{WorldObjectType::None};
    uint32_t m_transform_idx{0};
    vec3* m_pos_to_translate = nullptr;
    mat3* m_rot_to_rotate = nullptr;
    float* m_scale_to_scale = nullptr;
    bool* m_obj_dirty_marker = nullptr;

    cudaStream_t m_stream_id;

};

}