#include <neural-graphics-primitives/nerf_loader.h>

#include <synerfgine/cuda_helpers.h>
#include <synerfgine/engine.h>

#include <fmt/core.h>
#include <fstream>

// YOU CAN ONLY INCLUDE THIS ONCE IN THE WHOLE PROJECT
#include <zstr.hpp>

namespace sng {

json reload_network_from_file(const fs::path& path, bool& is_snapshot);
ngp::NerfDataset load_training_data(const fs::path& path, bool& is_training_data_available);

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

void Engine::load_file(const std::string& str_path) {
	fs::path path{str_path};
	if (path.exists()) {
		tlog::error() << "File '" << path.str() << "' does not exist.";
		return;
	}

    bool is_snapshot;
	nlohmann::json info = reload_network_from_file(path, is_snapshot);

	if (is_snapshot) {
		m_nerf_world.load_snapshot(info, path);
	} else {
		// is training data.
		try {
			// m_nerf_world.load_training_data(info);
		} catch (const std::exception& e) {
			tlog::error() << "Data at " << path.str() << "is neither snapshot nor training data.";
			return;
		}
	}
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

// returns the json
json reload_network_from_file(const filesystem::path& path, bool& is_snapshot) {
	namespace fs = filesystem;
	is_snapshot = equals_case_insensitive(path.extension(), "msgpack") || equals_case_insensitive(path.extension(), "ingp");

	if (!path.exists()) {
        std::string msg = "Network path does not exist: " + path.str();
		throw std::runtime_error(msg);
	}

	auto network_config_path{path};
	if (network_config_path.empty() || !network_config_path.exists()) {
		throw std::runtime_error{fmt::format("Network {} '{}' does not exist.", is_snapshot ? "snapshot" : "config", network_config_path.str())};
	}

	tlog::info() << "Loading network " << (is_snapshot ? "snapshot" : "config") << " from: " << network_config_path;

	json result{};
	if (is_snapshot) {
		std::ifstream f{native_string(network_config_path.str()), std::ios::in | std::ios::binary};
		if (equals_case_insensitive(network_config_path.extension(), "ingp")) {
			// zstr::ifstream applies zlib compression.
			zstr::istream zf{f};
			result = json::from_msgpack(zf);
		} else {
			result = json::from_msgpack(f);
		}
		// we assume parent pointers are already resolved in snapshots.
	} else if (equals_case_insensitive(network_config_path.extension(), "json")) {
		throw std::runtime_error{fmt::format("Loading network from json file {} not supported.", network_config_path.str())};
	}

	return result;
}

// returns the NerfDataset
ngp::NerfDataset load_training_data(const fs::path& path, bool& is_training_data_available) {
	if (!path.exists()) {
		throw std::runtime_error{fmt::format("Data path '{}' does not exist.", path.str())};
	}
	std::vector<filesystem::path> jsonpaths = {path.str()};
	auto nerf_t = ngp::load_nerf(jsonpaths);
	return nerf_t;
}

}
