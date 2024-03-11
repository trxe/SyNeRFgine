#pragma once

// #include <synerfgine/display.h>

#include <neural-graphics-primitives/render_buffer.h>
#include <neural-graphics-primitives/testbed.h>

#include <synerfgine/display.cuh>
#include <synerfgine/material.cuh>

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
    void init_buffers();
    void try_resize();
    void set_dead() { m_display.set_dead(); }
    void sync() { CUDA_CHECK_THROW(cudaStreamSynchronize(m_stream_id)); }
    Testbed::View& nerf_render_buffer_view() {
        return m_testbed->m_views.front();
    }

    bool m_is_dirty = true;
    Testbed* m_testbed;
    Display m_display;
    ivec2 m_next_frame_resolution;
    std::vector<Material> m_materials;

    std::vector<vec4> m_nerf_rgba_cpu;
    std::vector<float> m_nerf_depth_cpu;

    INIT_BENCHMARK();
	float m_render_ms = 0.0f;

    cudaStream_t m_stream_id;

};

}