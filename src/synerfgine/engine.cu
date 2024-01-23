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

void Engine::load_file(fs::path path) {
    m_nerf_world.load_network(path);
}

bool Engine::frame() {
    if (m_devices.empty()) {
        tlog::error("No CUDA devices found or attached.");
        return false;
    }

    auto& device = m_devices.front();
    if (!m_display.begin_frame(device, is_dirty)) return false;

    SyncedMultiStream synced_streams{m_stream.get(), 2};
    std::vector<std::future<void>> futures(2);
    auto render_buffer = m_display.get_render_buffer();
    render_buffer->set_color_space(ngp::EColorSpace::SRGB);
    render_buffer->set_tonemap_curve(ngp::ETonemapCurve::Identity);

    futures[0] = device.enqueue_task([this, &device, render_buffer, stream=synced_streams.get(0)]() {
        auto device_guard = use_device(stream, *render_buffer, device);
        m_syn_world.handle(device, m_display.get_window_res());
    });

    futures[1] = device.enqueue_task([this, &device, render_buffer, stream=synced_streams.get(1)]() {
        auto device_guard = use_device(stream, *render_buffer, device);
        m_nerf_world.handle(device, m_display.get_window_res());
    });

    for (auto& future : futures) {
        future.get();
    }

    {
        auto device_guard = use_device(synced_streams.get(0), *render_buffer, device);
        m_display.present(device, m_syn_world);
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