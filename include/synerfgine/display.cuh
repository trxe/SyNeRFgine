#pragma once

#include <neural-graphics-primitives/render_buffer.h>

#include <synerfgine/common.cuh>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_host.h>
#include <tiny-cuda-nn/common_device.h>

#ifdef NGP_GUI
#	include <imgui/backends/imgui_impl_glfw.h>
#	include <imgui/backends/imgui_impl_opengl3.h>
#	include <imgui/imgui.h>
#	include <imguizmo/ImGuizmo.h>
#	ifdef _WIN32
#		include <GL/gl3w.h>
#	else
#		include <GL/glew.h>
#	endif
#	include <GLFW/glfw3.h>
#	include <GLFW/glfw3native.h>
#	include <cuda_gl_interop.h>
#endif

#include <chrono>
#include <memory>

// Windows.h is evil
#undef min
#undef max
#undef near
#undef far

namespace sng {

using namespace tcnn;
using ngp::CudaRenderBuffer;

class Display {
public:
	Display() {}
	~Display() { destroy(); }
    GLFWwindow* init_window(int resw, int resh, const std::string& frag_fp);
	void destroy();
	void begin_frame();
	bool present( const vec3& clear_color, GLuint nerf_rgba_texid, GLuint nerf_depth_texid, GLuint syn_rgba_texid, GLuint syn_depth_texid, const ivec2& nerf_res, const ivec2& syn_res, const Foveation& fov, int filter_type); 
	bool is_alive() { return m_is_init; }
	void set_dead() { m_is_init = false; }

	const ivec2& get_window_res() const { return m_window_res; }
	void set_window_res(const ivec2& res)  { m_window_res = res; }

private:
	GLFWwindow* init_glfw(int resw, int resh);
	void init_imgui();
	void init_opengl_shaders(const std::string& frag_fp);
	void transfer_texture(const Foveation& foveation, [[maybe_unused]] GLint syn_rgba, GLint nerf_rgba, GLint rgba_filter_mode, 
		[[maybe_unused]] GLint syn_depth, GLint nerf_depth, GLint framebuffer, const ivec2& offset, const ivec2& resolution, const ivec2& nerf_res, const ivec2& syn_res, int filter_type);

	ivec2 m_window_res = ivec2(0);


	static bool m_is_init;

	GLFWwindow* m_glfw_window = nullptr;
	GLuint m_framebuffer = 0;
	GLuint m_blit_vao = 0;
	GLuint m_blit_program = 0;
};

}
