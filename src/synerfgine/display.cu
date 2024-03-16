#include <concurrencysal.h>
#include <memory>
#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/marching_cubes.h>

#include <tiny-cuda-nn/common.h>

#include <synerfgine/display.cuh>
#include <synerfgine/common.cuh>
#include <imgui/imgui.h>

namespace sng {

bool Display::m_is_init = false;

void glfw_error_callback(int error, const char* description) {
	tlog::error() << "GLFW error #" << error << ": " << description;
}

GLFWwindow* Display::init_window(int resw, int resh, const std::string& frag_fp) {
	if (m_is_init) return nullptr;
    // m_window_res = {resw, resh};
	m_glfw_window = init_glfw(resw, resh);
	init_opengl_shaders(frag_fp);
    init_imgui();
	Display::m_is_init = true;
	return m_glfw_window;
}

GLFWwindow* Display::init_glfw(int resw, int resh) {
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) {
        throw std::runtime_error{"GLFW could not be initialized."};
    }
    
    std::string title = "Synthetic Object NeRF Engine";
    m_glfw_window = glfwCreateWindow(resw, resh, title.c_str(), NULL, NULL);
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
	ini_filename = (File::get_root_dir()/"imgui.ini").str();
	// ini_filename = "./imgui.ini";
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

void Display::init_opengl_shaders(const std::string& frag_fp) {
	static const char* shader_vert = R"glsl(#version 140
		out vec2 UVs;
		void main() {
			UVs = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
			gl_Position = vec4(UVs * 2.0 - 1.0, 0.0, 1.0);
		})glsl";
	
	GLuint vert = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vert, 1, &shader_vert, NULL);
	glCompileShader(vert);
	ngp::check_shader(vert, "Blit vertex shader", false);

	std::string shader_frag_s = File::read_text(frag_fp);

	GLuint frag = glCreateShader(GL_FRAGMENT_SHADER);
	const GLchar *shader_frag[] = { shader_frag_s.c_str() };
	glShaderSource(frag, 1, shader_frag, NULL);
	glCompileShader(frag);
	ngp::check_shader(frag, "Blit fragment shader", false);

	m_blit_program = glCreateProgram();
	glAttachShader(m_blit_program, vert);
	glAttachShader(m_blit_program, frag);
	glLinkProgram(m_blit_program);
	ngp::check_shader(m_blit_program, "Blit shader program", true);

	glDeleteShader(vert);
	glDeleteShader(frag);

	glGenVertexArrays(1, &m_blit_vao);
}


void Display::begin_frame() {
	if (glfwWindowShouldClose(m_glfw_window) || ImGui::IsKeyPressed(GLFW_KEY_ESCAPE) || ImGui::IsKeyPressed(GLFW_KEY_Q)) {
		destroy();
		return;
	}

	glfwPollEvents();

	// UI begin
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
	ImGuizmo::BeginFrame();
}

void Display::transfer_texture(const Foveation& foveation, [[maybe_unused]] GLint syn_rgba, GLint syn_depth, GLint rgba_filter_mode, 
	GLint nerf_rgba, GLint nerf_depth, GLint framebuffer, const ivec2& offset, const ivec2& resolution) {
	if (m_blit_program == 0) {
		return;
	}

	bool tex = glIsEnabled(GL_TEXTURE_2D);
	bool depth = glIsEnabled(GL_DEPTH_TEST);
	bool cull = glIsEnabled(GL_CULL_FACE);

	if (!tex) 
		glEnable(GL_TEXTURE_2D);
	if (!depth) 
		glEnable(GL_DEPTH_TEST);
	if (cull) 
		glDisable(GL_CULL_FACE);

	glDepthFunc(GL_ALWAYS);
	glDepthMask(GL_TRUE);

	glBindVertexArray(m_blit_vao);
	glUseProgram(m_blit_program);
	auto syn_rgba_uniform = glGetUniformLocation(m_blit_program, "syn_rgba");
	auto syn_depth_uniform = glGetUniformLocation(m_blit_program, "syn_depth");
	glUniform1i(syn_rgba_uniform, 0);
	glUniform1i(syn_depth_uniform, 1);
	auto nerf_rgba_uniform = glGetUniformLocation(m_blit_program, "nerf_rgba");
	auto nerf_depth_uniform = glGetUniformLocation(m_blit_program, "nerf_depth");
	glUniform1i(nerf_rgba_uniform, 2);
	glUniform1i(nerf_depth_uniform, 3);

	auto bind_warp = [&](const ngp::FoveationPiecewiseQuadratic& warp, const std::string& uniform_name) {
		glUniform1f(glGetUniformLocation(m_blit_program, (uniform_name + ".al").c_str()), warp.al);
		glUniform1f(glGetUniformLocation(m_blit_program, (uniform_name + ".bl").c_str()), warp.bl);
		glUniform1f(glGetUniformLocation(m_blit_program, (uniform_name + ".cl").c_str()), warp.cl);

		glUniform1f(glGetUniformLocation(m_blit_program, (uniform_name + ".am").c_str()), warp.am);
		glUniform1f(glGetUniformLocation(m_blit_program, (uniform_name + ".bm").c_str()), warp.bm);

		glUniform1f(glGetUniformLocation(m_blit_program, (uniform_name + ".ar").c_str()), warp.ar);
		glUniform1f(glGetUniformLocation(m_blit_program, (uniform_name + ".br").c_str()), warp.br);
		glUniform1f(glGetUniformLocation(m_blit_program, (uniform_name + ".cr").c_str()), warp.cr);

		glUniform1f(glGetUniformLocation(m_blit_program, (uniform_name + ".switch_left").c_str()), warp.switch_left);
		glUniform1f(glGetUniformLocation(m_blit_program, (uniform_name + ".switch_right").c_str()), warp.switch_right);

		glUniform1f(glGetUniformLocation(m_blit_program, (uniform_name + ".inv_switch_left").c_str()), warp.inv_switch_left);
		glUniform1f(glGetUniformLocation(m_blit_program, (uniform_name + ".inv_switch_right").c_str()), warp.inv_switch_right);
	};

	bind_warp(foveation.warp_x, "warp_x");
	bind_warp(foveation.warp_y, "warp_y");

	glActiveTexture(GL_TEXTURE3);
	glBindTexture(GL_TEXTURE_2D, nerf_depth);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, rgba_filter_mode);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, rgba_filter_mode);

	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, nerf_rgba);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, syn_depth);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, syn_rgba);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, rgba_filter_mode);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, rgba_filter_mode);

	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
	glViewport(offset.x, offset.y, resolution.x, resolution.y);

	glDrawArrays(GL_TRIANGLES, 0, 3);

	glBindVertexArray(0);
	glUseProgram(0);

	glDepthFunc(GL_LESS);

	// restore old state
	if (!tex) glDisable(GL_TEXTURE_2D);
	if (!depth) glDisable(GL_DEPTH_TEST);
	if (cull) glEnable(GL_CULL_FACE);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

bool Display::present(GLuint nerf_rgba_texid, GLuint nerf_depth_texid, GLuint syn_rgba_texid, GLuint syn_depth_texid, const ivec2& nerf_extent, const Foveation& fov) {
	if (!m_glfw_window) {
		throw std::runtime_error{"Window must be initialized to be presented."};
	}
	// UI DRAWING
	CUDA_CHECK_THROW(cudaDeviceSynchronize());
	glfwGetFramebufferSize(m_glfw_window, &m_window_res.x, &m_window_res.y);

	// IMAGE RENDER
	glViewport(0, 0, m_window_res.x, m_window_res.y);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// glEnable(GL_BLEND);
	// glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
	// glBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

    ivec2 extent = {(int)((float)m_window_res.x / nerf_extent.x), (int)((float)m_window_res.y / nerf_extent.y)};
	ivec2 top_left{0, m_window_res.y - extent.y};
	transfer_texture(fov, syn_rgba_texid, syn_depth_texid, GL_LINEAR, nerf_rgba_texid, nerf_depth_texid, m_framebuffer, top_left, extent);
	glFinish();

	// IMGUI
	ImDrawList* list = ImGui::GetBackgroundDrawList();
	list->AddCallback(ImDrawCallback_ResetRenderState, nullptr);

	// Visualizations are only meaningful when rendering a single view
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

	glfwSwapBuffers(m_glfw_window);

	// Make sure all the OGL code finished its business here.
	// Any code outside of this function needs to be able to freely write to
	// textures without being worried about interfering with rendering.
	glFinish();
	return true;
}

void Display::destroy() {
	if (!Display::m_is_init) {
		return;
	}

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	glfwDestroyWindow(m_glfw_window);
	glfwTerminate();

	m_glfw_window = nullptr;
	m_is_init = false;
}

}