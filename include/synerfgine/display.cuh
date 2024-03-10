#pragma once

#include <neural-graphics-primitives/render_buffer.h>

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_host.h>

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
using ngp::GLTexture;

// class Renderer {
// public:
// 	bool begin_frame(const ivec2& window_res);
// 	bool present(const ivec2& m_n_views, std::shared_ptr<ngp::GLTexture> rgba, std::shared_ptr<ngp::GLTexture> depth, CudaDevice& device);
// 	bool present(const ivec2& m_n_views, std::shared_ptr<ngp::GLTexture> syn_rgba, std::shared_ptr<ngp::GLTexture> syn_depth, const CudaRenderBufferView& syn_view,
// 		std::shared_ptr<ngp::GLTexture> nerf_rgba, std::shared_ptr<ngp::GLTexture> nerf_depth, const CudaRenderBufferView& nerf_view,CudaDevice& device);
// 	void end_frame();
// private:
// 	void blit_texture(const ngp::Foveation& foveation, GLint syn_rgba, GLint nerf_rgba, GLint rgba_filter_mode, 
// 		GLint syn_depth, GLint nerf_depth, GLint framebuffer, const ivec2& offset, const ivec2& resolution);
// 	GLFWwindow* m_glfw_window = nullptr;
// 	ivec2 m_window_res = ivec2(0);

// 	// The VAO will be empty, but we need a valid one for attribute-less rendering
// 	GLuint m_blit_vao = 0;
// 	GLuint m_blit_program = 0;

// 	std::vector<vec4> m_cpu_frame_buffer_syn; 
// 	std::vector<float> m_cpu_depth_buffer_syn; 
// 	std::vector<vec4> m_cpu_frame_buffer_nerf; 
// 	std::vector<float> m_cpu_depth_buffer_nerf; 

// 	void init_opengl_shaders();
// };

class Display {
public:
	Display() {}
	~Display() { destroy(); }
    GLFWwindow* init_window(int resw, int resh);
	void destroy();
	void resize(const ivec2& window_res);
	void begin_frame(bool& is_dirty);
	// bool present(); 
	void end_frame();

	ivec2 get_window_res() const {
		return m_window_res;
	}

private:
	GLFWwindow* init_glfw();
	void init_buffers();
	void init_imgui();
	// void copy_textures(std::shared_ptr<ngp::GLTexture> rgba, 
		// std::shared_ptr<ngp::GLTexture> depth, CudaDevice& device);

	ivec2 m_window_res = ivec2(0);

	// Buffers
	std::shared_ptr<ngp::GLTexture> m_rgba_render_texture;
	std::shared_ptr<ngp::GLTexture> m_depth_render_texture;
	std::shared_ptr<ngp::CudaRenderBuffer> m_render_buffer;

	static bool m_is_init;

	// Metrics
	std::chrono::system_clock::time_point m_last_timestamp = std::chrono::system_clock::now();
	float m_last_frame_time = 0.000001f;

	GLFWwindow* m_glfw_window = nullptr;
};

}
