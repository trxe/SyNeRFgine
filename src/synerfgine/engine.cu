#include <synerfgine/cuda_helpers.h>
#include <synerfgine/engine.h>

namespace sng {

Engine::Engine() {
    m_devices.emplace_back(find_cuda_device(), true);
}

void Engine::init(int res_width, int res_height) {
    GLFWwindow* glfw_window = m_display.init_window(res_width, res_height, false);
    glfwSetWindowUserPointer(glfw_window, this);
	glfwSetWindowSizeCallback(glfw_window, [](GLFWwindow* window, int width, int height) {
		Engine* engine = (Engine*)glfwGetWindowUserPointer(window);
		if (engine) {
			engine->redraw_next_frame();
		}
	});
}

bool Engine::frame() {
    if (m_devices.empty()) {
        tlog::error("No CUDA devices found or attached.");
        return false;
    }

    auto& device = m_devices.front();
    if (!m_display.begin_frame(device, is_dirty)) return false;

    SyncedMultiStream synced_streams{m_stream.get(), 1};
    std::vector<std::future<void>> futures(1);
    auto render_buffer = m_display.get_render_buffer();
    render_buffer->set_color_space(ngp::EColorSpace::SRGB);
    render_buffer->set_tonemap_curve(ngp::ETonemapCurve::Identity);

    futures[0] = device.enqueue_task([this, &device, render_buffer, stream=synced_streams.get(0)]() {
        auto device_guard = use_device(stream, *render_buffer, device);
        m_syn_world.handle(device, m_display.get_window_res());
        m_display.present(device, m_syn_world);
        m_display.end_frame();
    });

    if (futures[0].valid()) {
        futures[0].get();
   }

    return true;
}

Engine::~Engine() {
    for (auto&& device : m_devices) {
        device.clear();
    }
}

}