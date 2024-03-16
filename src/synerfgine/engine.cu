#include <synerfgine/engine.cuh>
#include <synerfgine/common.cuh>
#include <filesystem/path.h>
#include <iostream>
#include <type_traits>

namespace sng {

void Engine::set_virtual_world(const std::string& config_fp) {
    nlohmann::json config = File::read_json(config_fp);
    nlohmann::json& mat_conf = config["materials"];
    for (uint32_t i = 0; i < mat_conf.size(); ++i) {
        m_materials.emplace_back(i, mat_conf[i]);
    }
    nlohmann::json& obj_conf = config["objfile"];
    for (uint32_t i = 0; i < obj_conf.size(); ++i) {
        m_objects.emplace_back(i, obj_conf[i]);
    }
    nlohmann::json& light_conf = config["lights"];
    for (uint32_t i = 0; i < light_conf.size(); ++i) {
        m_lights.emplace_back(i, light_conf[i]);
    }
	m_raytracer.set_objs(m_objects);
}

void Engine::init(int res_width, int res_height, const std::string& frag_fp, Testbed* nerf) {
	m_testbed = nerf;
    m_testbed->m_train = false;
    m_testbed->set_n_views(1);
    m_next_frame_resolution = {res_width, res_height};
    GLFWwindow* glfw_window = m_display.init_window(res_width, res_height, frag_fp);
    glfwSetWindowUserPointer(glfw_window, this);
	glfwSetWindowSizeCallback(glfw_window, [](GLFWwindow* window, int width, int height) {
		Engine* engine = (Engine*)glfwGetWindowUserPointer(window);
		if (engine) {
            engine->m_next_frame_resolution = {width, height};
			engine->redraw_next_frame();
		}
	});
    glfwSetWindowCloseCallback(glfw_window, [](GLFWwindow* window) {
		Engine* engine = (Engine*)glfwGetWindowUserPointer(window);
		if (engine) { engine->set_dead(); }
	});
    Testbed::CudaDevice& device = m_testbed->primary_device();
    m_stream_id = device.stream();
}

void Engine::try_resize() {
    ivec2 curr_window_res = m_display.get_window_res();
    if (curr_window_res != m_next_frame_resolution || !m_testbed->m_render_skip_due_to_lack_of_camera_movement_counter || 
            m_last_target_fps != m_testbed->m_dynamic_res_target_fps) {
        m_display.set_window_res(m_next_frame_resolution);
        m_testbed->m_window_res = m_next_frame_resolution;
        auto& view = nerf_render_buffer_view();
        m_last_target_fps = m_testbed->m_dynamic_res_target_fps;
		float factor = 5.0f / m_testbed->m_dynamic_res_target_fps;
        // tlog::success() << "Scaling full resolution by " << factor;
        auto new_res = downscale_resolution(m_next_frame_resolution, factor);
        view.resize(new_res);
        sync(m_stream_id);

        m_raytracer.enlarge(m_next_frame_resolution);
        // m_raytracer.enlarge(new_res);
    }
}

void Engine::imgui() {
    if (ImGui::Begin("Synthetic World")) {
        if (ImGui::CollapsingHeader("Materials", ImGuiTreeNodeFlags_DefaultOpen)) {
            for (auto& m : m_materials) { m.imgui(); }
        }
        if (ImGui::CollapsingHeader("Objects", ImGuiTreeNodeFlags_DefaultOpen)) {
            for (auto& m : m_objects) { m.imgui(); }
        }
        if (ImGui::CollapsingHeader("Lights", ImGuiTreeNodeFlags_DefaultOpen)) {
            for (auto& m : m_lights) { m.imgui(); }
        }
        m_raytracer.imgui();
    }
    ImGui::End();
}

bool Engine::frame() {
    if (!m_display.is_alive()) return false;
    Testbed::CudaDevice& device = m_testbed->primary_device();
    device.device_guard();
	m_display.begin_frame();
    try_resize();
    sync(m_stream_id);
    m_testbed->handle_user_input();
    imgui();
	ImDrawList* list = ImGui::GetBackgroundDrawList();
    // m_testbed->draw_visualizations(list, m_testbed->m_smoothed_camera);

    m_testbed->apply_camera_smoothing(__timer.get_ave_time("nerf"));

    auto& view = nerf_render_buffer_view();

    auto nerf_view = view.render_buffer->view();
    __timer.reset();
    {
        sync(m_stream_id);
        m_testbed->render( m_stream_id, view );
        sync(m_stream_id);
        view.prev_camera = view.camera0;
        view.prev_foveation = view.foveation;

        ivec2 nerf_res = nerf_view.resolution;
        auto n_elements = product(nerf_res);
    }
    m_render_ms = (float)__timer.log_time("nerf");
    m_testbed->m_frame_ms.set(m_render_ms);
    m_testbed->m_rgba_render_textures.front()->load_gpu(nerf_view.frame_buffer, nerf_view.resolution, m_nerf_rgba_cpu);
    m_testbed->m_depth_render_textures.front()->load_gpu(nerf_view.depth_buffer, nerf_view.resolution, 1, m_nerf_depth_cpu);
    GLuint nerf_rgba_texid = m_testbed->m_rgba_render_textures.front()->texture();
    GLuint nerf_depth_texid = m_testbed->m_depth_render_textures.front()->texture();

    {
        vec2 focal_length = m_testbed->calc_focal_length(
            m_raytracer.resolution(),
            m_testbed->m_relative_focal_length, 
            m_testbed->m_fov_axis, 
            m_testbed->m_zoom);
        vec2 screen_center = m_testbed->render_screen_center(view.screen_center);
        m_raytracer.render(
            m_materials, 
            m_objects,
            m_lights,
            view, 
            screen_center,
            nerf_view.spp,
            focal_length,
            m_testbed->m_snap_to_pixel_centers,
            m_testbed->m_nerf.density_grid_bitfield.data()
        );
    }
    m_raytracer.load(m_syn_rgba_cpu, m_syn_depth_cpu);
    GLuint syn_rgba_texid = m_raytracer.m_rgba_texture->texture();
    GLuint syn_depth_texid = m_raytracer.m_depth_texture->texture();
    m_display.present(nerf_rgba_texid, nerf_depth_texid, syn_rgba_texid, syn_depth_texid, m_testbed->m_n_views(0), view.foveation);
    return m_display.is_alive();
}

}