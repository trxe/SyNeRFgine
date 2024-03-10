#include <memory>
#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/marching_cubes.h>

#include <tiny-cuda-nn/common.h>

#include <synerfgine/display.cuh>
#include <imgui/imgui.h>

namespace sng {

bool Display::m_is_init = false;

void glfw_error_callback(int error, const char* description) {
	tlog::error() << "GLFW error #" << error << ": " << description;
}

GLFWwindow* Display::init_window(int resw, int resh) {
	if (m_is_init) return nullptr;
    m_window_res = {resw, resh};
	m_glfw_window = init_glfw();
    init_imgui();
	init_buffers();
	Display::m_is_init = true;
	return m_glfw_window;
}

GLFWwindow* Display::init_glfw() {
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) {
        throw std::runtime_error{"GLFW could not be initialized."};
    }
    
    std::string title = "Synthetic Object NeRF Engine";
    m_glfw_window = glfwCreateWindow(m_window_res.x, m_window_res.y, title.c_str(), NULL, NULL);
    if (m_glfw_window == NULL) {
        throw std::runtime_error{"GLFW window could not be created."};
    }
    glfwMakeContextCurrent(m_glfw_window);
#ifdef _WIN32
    if (gl3wInit()) {
        throw std::runtime_error{"GL3W could not be initialized."};
    }
#else
    glewExperimental = 1;
    if (glewInit()) {
        throw std::runtime_error{"GLEW could not be initialized."};
    }
#endif
    glfwSwapInterval(0); // Disable vsync

    GLint gl_version_minor, gl_version_major;
    glGetIntegerv(GL_MINOR_VERSION, &gl_version_minor);
    glGetIntegerv(GL_MAJOR_VERSION, &gl_version_major);

    if (gl_version_major < 3 || (gl_version_major == 3 && gl_version_minor < 1)) {
        throw std::runtime_error{fmt::format("Unsupported OpenGL version {}.{}. instant-ngp requires at least OpenGL 3.1", gl_version_major, gl_version_minor)};
    }

    tlog::success() << "Initialized OpenGL version " << glGetString(GL_VERSION);

	// init_opengl_shaders();

	return m_glfw_window;
}

void Display::init_imgui() {
	float xscale, yscale;
	glfwGetWindowContentScale(m_glfw_window, &xscale, &yscale);

	// IMGUI init
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;

	// By default, imgui places its configuration (state of the GUI -- size of windows,
	// which regions are expanded, etc.) in ./imgui.ini relative to the working directory.
	// Instead, we would like to place imgui.ini in the directory that instant-ngp project
	// resides in.
	static std::string ini_filename;
	// ini_filename = (Utils::get_root_dir()/"imgui.ini").string();
	ini_filename = "./imgui.ini";
	io.IniFilename = ini_filename.c_str();

	// New ImGui event handling seems to make camera controls laggy if input trickling is true.
	// So disable input trickling.
	io.ConfigInputTrickleEventQueue = false;
	ImGui::StyleColorsDark();
	ImGui_ImplGlfw_InitForOpenGL(m_glfw_window, true);
	ImGui_ImplOpenGL3_Init("#version 140");

	ImGui::GetStyle().ScaleAllSizes(xscale);
	ImFontConfig font_cfg;
	font_cfg.SizePixels = 13.0f * xscale;
	io.Fonts->AddFontDefault(&font_cfg);
}

// void Renderer::init_opengl_shaders(const std::string& vert_fp, const std::string& frag_fp) {
	
// 	GLuint vert = glCreateShader(GL_VERTEX_SHADER);
// 	glShaderSource(vert, 1, &shader_vert, NULL);
// 	glCompileShader(vert);
// 	ngp::check_shader(vert, "Blit vertex shader", false);

// 	GLuint frag = glCreateShader(GL_FRAGMENT_SHADER);
// 	glShaderSource(frag, 1, &shader_frag, NULL);
// 	glCompileShader(frag);
// 	ngp::check_shader(frag, "Blit fragment shader", false);

// 	m_blit_program = glCreateProgram();
// 	glAttachShader(m_blit_program, vert);
// 	glAttachShader(m_blit_program, frag);
// 	glLinkProgram(m_blit_program);
// 	ngp::check_shader(m_blit_program, "Blit shader program", true);

// 	glDeleteShader(vert);
// 	glDeleteShader(frag);

// 	glGenVertexArrays(1, &m_blit_vao);
// }

void Display::init_buffers() {
	// Make sure there's at least one usable render texture
	m_rgba_render_texture = std::make_shared<GLTexture>();
	m_depth_render_texture = std::make_shared<GLTexture>();

	m_render_buffer = std::make_shared<CudaRenderBuffer>(m_rgba_render_texture, m_depth_render_texture);
	m_render_buffer->resize(m_window_res);
	m_render_buffer->disable_dlss();
}

void Display::begin_frame(bool& is_dirty) {
	if (glfwWindowShouldClose(m_glfw_window) || ImGui::IsKeyPressed(GLFW_KEY_ESCAPE) || ImGui::IsKeyPressed(GLFW_KEY_Q)) {
		destroy();
		return;
	}

	glfwPollEvents();
	glfwGetFramebufferSize(m_glfw_window, &m_window_res.x, &m_window_res.y);
	if (is_dirty) {
		m_render_buffer->resize(m_window_res);
		is_dirty = false;
	}

	// UI begin
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
	ImGuizmo::BeginFrame();
}

void Display::end_frame() {
	auto time_now = std::chrono::system_clock::now();
	m_last_frame_time = (float)std::chrono::duration_cast<std::chrono::milliseconds>(time_now - m_last_timestamp).count();
	m_last_timestamp = time_now;
 }


// bool Renderer::present(const ivec2& m_n_views, std::shared_ptr<ngp::GLTexture> syn_rgba, std::shared_ptr<ngp::GLTexture> syn_depth, const ngp::CudaRenderBufferView& syn_view,
// 		std::shared_ptr<ngp::GLTexture> nerf_rgba, std::shared_ptr<ngp::GLTexture> nerf_depth, const CudaRenderBufferView& nerf_view, CudaDevice& device) { 
// 	// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#opengl-interoperability
// 	if (!m_glfw_window) {
// 		throw std::runtime_error{"Window must be initialized to be presented."};
// 	}
// 	// UI DRAWING
// 	glViewport(0, 0, display_w, display_h);

// 	ImDrawList* list = ImGui::GetBackgroundDrawList();
// 	list->AddCallback(ImDrawCallback_ResetRenderState, nullptr);

// 	// Visualizations are only meaningful when rendering a single view
// 	ImGui::Render();
// 	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

// 	glfwSwapBuffers(m_glfw_window);

// 	// Make sure all the OGL code finished its business here.
// 	// Any code outside of this function needs to be able to freely write to
// 	// textures without being worried about interfering with rendering.
// 	glFinish();

// 	return true;
// }

void Display::destroy() {
	if (!Display::m_is_init) {
		return;
	}

	m_render_buffer = nullptr;

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	glfwDestroyWindow(m_glfw_window);
	glfwTerminate();

	m_glfw_window = nullptr;
	m_is_init = false;
}

}