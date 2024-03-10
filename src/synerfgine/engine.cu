#include <synerfgine/engine.cuh>
#include <synerfgine/common.cuh>
#include <filesystem/path.h>
#include <iostream>

namespace sng {

void Engine::set_virtual_world(const std::string& config_fp) {
    nlohmann::json config = File::read_json(config_fp);
    nlohmann::json& mat_conf = config["materials"];
    for (uint32_t i = 0; i < mat_conf.size(); ++i) {
        m_materials.emplace_back(i, mat_conf[i]);
    }
    for (const auto& m: m_materials ) {
        m.print();
    }
    // init_mat(mat_conf);
    tlog::success() << "SETTING VIRTULA WORLD : " << config_fp << " with mat count " << m_materials.size() << " " << mat_conf.size();
    nlohmann::json& obj_conf = config["objfile"];
    // init_objs(obj_conf);
}

void Engine::init(int res_width, int res_height, const std::string& frag_fp, Testbed* nerf) {
    // GLFWwindow* glfw_window = m_display.init_window(res_width, res_height, false);
    // glfwSetWindowUserPointer(glfw_window, this);
	// glfwSetWindowSizeCallback(glfw_window, [](GLFWwindow* window, int width, int height) {
	// 	Engine* engine = (Engine*)glfwGetWindowUserPointer(window);
	// 	if (engine) {
	// 		engine->redraw_next_frame();
	// 	}
	// });
	m_testbed = nerf;
    m_testbed->m_train = false;
    m_display.init_window(res_width, res_height);
}

bool Engine::frame() {
	m_display.begin_frame(m_is_dirty);
	m_display.end_frame();
    return true;
}

}