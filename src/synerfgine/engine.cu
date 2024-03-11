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
    for (const auto& m: m_materials) { m.print(); }
    nlohmann::json& obj_conf = config["objfile"];
    for (uint32_t i = 0; i < obj_conf.size(); ++i) {
        m_objects.emplace_back(i, obj_conf[i]);
    }
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
    if (curr_window_res != m_next_frame_resolution || !m_testbed->m_render_skip_due_to_lack_of_camera_movement_counter) {
        m_display.set_window_res(m_next_frame_resolution);
        m_testbed->m_window_res = m_next_frame_resolution;
        auto& view = nerf_render_buffer_view();
        auto nerf_view = view.render_buffer->view();
        uint32_t nerf_res = product(nerf_view.resolution);
        uint32_t n_pixels_full_res = product(curr_window_res);
		float pixel_ratio = ((float)nerf_res / (float)n_pixels_full_res);
		float last_factor = std::sqrt(pixel_ratio);
		float factor = std::sqrt(pixel_ratio / m_render_ms * 1000.0f / m_testbed->m_dynamic_res_target_fps);
        auto new_res = downscale_resolution(m_next_frame_resolution, factor);
        view.resize(new_res);
        sync(m_stream_id);

        m_raytracer.enlarge(new_res);
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
    m_testbed->apply_camera_smoothing(__timer.get_ave_time("nerf"));

    auto& view = nerf_render_buffer_view();

    auto nerf_view = view.render_buffer->view();
	vec2 focal_length = m_testbed->calc_focal_length(
        nerf_view.resolution, 
        m_testbed->m_relative_focal_length, 
        m_testbed->m_fov_axis, 
        m_testbed->m_zoom);
	vec2 screen_center = m_testbed->render_screen_center(view.screen_center);
    __timer.reset();
    {
        sync(m_stream_id);
        m_testbed->primary_device().set_render_buffer_view(nerf_view);
        if (!m_testbed->m_render_skip_due_to_lack_of_camera_movement_counter) {
            m_testbed->reset_accumulation(false);
            nerf_view.clear(m_stream_id);
        }
        m_testbed->render_frame(
            m_stream_id,
            view.camera0,
            view.camera1,
            view.prev_camera,
            screen_center,
            view.relative_focal_length,
            view.rolling_shutter,
            view.foveation,
            view.prev_foveation,
            view.visualized_dimension,
            *view.render_buffer
        );
        sync(m_stream_id);
        view.prev_camera = view.camera0;
        view.prev_foveation = view.foveation;

        ivec2 nerf_res = nerf_view.resolution;
        auto n_elements = product(nerf_res);
        // linear_kernel(debug_shade, 0, m_stream_id, n_elements, nerf_view.frame_buffer, vec3(1.0, 0.0, 0.0), nerf_view.depth_buffer, 0.0);
        // linear_kernel(print_shade, 0, m_stream_id, n_elements, nerf_view.frame_buffer, nerf_view.depth_buffer);
        // sync();
    }
    m_render_ms = (float)__timer.log_time("nerf");
    m_testbed->m_frame_ms.set(m_render_ms);
    m_testbed->m_rgba_render_textures.front()->load_gpu(nerf_view.frame_buffer, nerf_view.resolution, m_nerf_rgba_cpu);
    m_testbed->m_depth_render_textures.front()->load_gpu(nerf_view.depth_buffer, nerf_view.resolution, 1, m_nerf_depth_cpu);
    GLuint nerf_rgba_texid = m_testbed->m_rgba_render_textures.front()->texture();
    GLuint nerf_depth_texid = m_testbed->m_depth_render_textures.front()->texture();

    m_raytracer.render(
        m_materials, 
        m_objects,
        view, 
        screen_center,
        nerf_view.spp,
        focal_length,
        m_testbed->m_snap_to_pixel_centers
    );
    m_raytracer.load(m_syn_rgba_cpu, m_syn_depth_cpu);
    GLuint syn_rgba_texid = m_raytracer.m_rgba_texture->texture();
    GLuint syn_depth_texid = m_raytracer.m_depth_texture->texture();

	ImDrawList* list = ImGui::GetBackgroundDrawList();
    m_testbed->draw_visualizations(list, m_testbed->m_smoothed_camera);
    m_display.present(nerf_rgba_texid, nerf_depth_texid, syn_rgba_texid, syn_depth_texid, m_testbed->m_n_views(0), view.foveation);

    return m_display.is_alive();
}

}