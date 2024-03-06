#include <synerfgine/engine.h>
#include <iostream>

namespace sng {

Engine::Engine() {
    m_devices.emplace_back(find_cuda_device(), true);
}

void Engine::init(int res_width, int res_height, Testbed* nerf) {
    GLFWwindow* glfw_window = m_display.init_window(res_width, res_height, false);
    glfwSetWindowUserPointer(glfw_window, this);
	glfwSetWindowSizeCallback(glfw_window, [](GLFWwindow* window, int width, int height) {
		Engine* engine = (Engine*)glfwGetWindowUserPointer(window);
		if (engine) {
			engine->redraw_next_frame();
		}
	});
    m_nerf_world.init(nerf);
    m_syn_world.mut_camera().set_default_matrix(nerf->m_camera);
	m_testbed = nerf;
    m_testbed->m_train = false;
}

void Engine::debug_rays() {
    auto& device = m_devices.front();
    m_syn_world.handle_user_input(m_display.get_window_res());
    std::shared_ptr<CudaRenderBuffer> render_buffer = m_syn_world.render_buffer();
    render_buffer->set_color_space(ngp::EColorSpace::SRGB);
    render_buffer->set_tonemap_curve(ngp::ETonemapCurve::Identity);
    if (m_show_nerf) {
        m_nerf_world.debug_init_rays(device, m_display.get_window_res(), m_syn_world.camera());
        m_syn_world.reset_frame(device, m_display.get_window_res());
    } else {
        m_syn_world.debug_init_rays(device, m_display.get_window_res());
        m_nerf_world.reset_frame(device, m_display.get_window_res());
    }
}

bool Engine::frame() {
    if (m_devices.empty()) {
        tlog::error("No CUDA devices found or attached.");
        return false;
    }

    auto& device = m_devices.front();
    is_dirty = m_syn_world.handle_user_input(m_display.get_window_res());

    if (!m_display.begin_frame(device, is_dirty)) return false;

    // ImGui::Begin("Debug");
    // if (ImGui::CollapsingHeader("Toggle Buffer Views", ImGuiTreeNodeFlags_DefaultOpen)) {
    //     if (ImGui::RadioButton("Show ray dir buffers", m_is_debug_rays)) {
    //         m_is_debug_rays = !m_is_debug_rays;
    //     }
    //     if (m_is_debug_rays && ImGui::RadioButton("Rays Nerf (Syn when off)", m_show_nerf)) {
    //         m_show_nerf = !m_show_nerf;
    //     }
    // }
    // if (m_is_debug_rays) { 
    //     debug_rays(); 
    //     ImGui::End();
    //     m_display.present(device, m_syn_world, m_nerf_world);
    //     m_display.end_frame();
    //     return true;
    // }

    // ImGui::End();

    {
        SyncedMultiStream synced_streams{m_stream.get(), 3};
        std::vector<std::future<void>> futures(3);

        futures[0] = device.enqueue_task([this, &device, stream=synced_streams.get(0)]() {
            std::shared_ptr<CudaRenderBuffer> render_buffer = m_syn_world.render_buffer();
            render_buffer->set_color_space(ngp::EColorSpace::SRGB);
            render_buffer->set_tonemap_curve(ngp::ETonemapCurve::Identity);
            m_syn_world.handle(device, m_display.get_window_res());
            m_syn_world.shoot_network(device, m_display.get_window_res(), *m_testbed);
            m_syn_world.debug_visualize_pos(device, m_syn_world.sun_pos(), vec3(0.0f, 1.0f, 0.0f), 0.2f);
        });

        futures[1] = device.enqueue_task([this, &device, stream=synced_streams.get(1)]() {
            std::shared_ptr<CudaRenderBuffer> render_buffer = m_nerf_world.render_buffer();
            auto device_guard = use_device(stream, *render_buffer, device);
            m_nerf_world.handle(device, m_syn_world.camera(), 
                m_syn_world.sun(), 
                m_syn_world.get_object(), 
                m_display.get_window_res());
        });

        if (futures[0].valid()) {
            futures[0].get();
        }

        if (futures[1].valid()) {
            futures[1].get(); 
        }

        m_display.present(device, m_syn_world, m_nerf_world);
        m_display.end_frame();
    }

    return true;
}

Engine::~Engine() {
    for (auto&& device : m_devices) {
        device.clear();
    }
}

}