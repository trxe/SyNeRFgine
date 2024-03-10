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
    bool m_is_dirty = true;
    Testbed* m_testbed;
    Display m_display;
    std::vector<Material> m_materials;
};

}