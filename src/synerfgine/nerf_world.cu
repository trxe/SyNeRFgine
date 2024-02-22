#include <synerfgine/cuda_helpers.h>
#include <synerfgine/nerf_world.h>

#include <tiny-cuda-nn/common.h>

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

namespace sng {
constexpr bool TRAIN_WITHOUT_RENDER = true;

namespace fs = std::filesystem;
using namespace tcnn;
using ngp::Testbed;
using ngp::GLTexture;

__global__ void debug_depth(const uint32_t n_elements, vec4* __restrict__ rgba, float* __restrict__ depth);

inline ivec2 downscale_resolution(const ivec2& resolution, float scale) {
    return clamp(ivec2(vec2(resolution) * scale), resolution / 16, resolution);
}

NerfWorld::NerfWorld() {
	m_rgba_render_textures = std::make_shared<GLTexture>();
	m_depth_render_textures = std::make_shared<GLTexture>();

	m_render_buffer = std::make_shared<CudaRenderBuffer>(m_rgba_render_textures, m_depth_render_textures);
    m_render_buffer_view = m_render_buffer->view();
	m_render_buffer->disable_dlss();
}

void NerfWorld::init(Testbed* testbed) {
    m_testbed = testbed;
}

bool NerfWorld::handle(sng::CudaDevice& device, const Camera& cam, const Light& sun, std::optional<VirtualObject>& vo, const ivec2& resolution) {
    if (!m_testbed) return false;
    auto cam_matrix = cam.get_matrix();
    mat4 vo_matrix = vo.has_value() ? vo.value().get_transform() : mat4::identity();
    if (m_last_camera == cam_matrix && m_last_sun == sun && vo_matrix == m_last_vo && !m_is_dirty) return false;

    constexpr float pixel_ratio = 1.0f;
    float factor = std::sqrt(pixel_ratio / m_render_ms * 1000.0f / m_dynamic_res_target_fps);
    factor = 8.f / (float)m_fixed_res_factor;
    factor = clamp(factor, 1.0f / 16.0f, 1.0f);

    auto new_resolution = downscale_resolution(resolution, factor);
    if (new_resolution != m_resolution) {
        m_testbed->m_nerf.training.dataset.scale = 1.0;
        m_resolution = new_resolution;
        m_rgba_render_textures->resize(m_resolution, 4);
        m_depth_render_textures->resize(m_resolution, 1);
        m_render_buffer->resize(m_resolution);
        m_render_buffer->set_hidden_area_mask(nullptr);
        m_render_buffer->disable_dlss();
        m_render_buffer_view = m_render_buffer->view();
    }
    auto stream = device.stream();
    auto& testbed_device = m_testbed->primary_device();
    testbed_device.set_render_buffer_view(m_render_buffer_view);
    // MUST use m_testbed's sync_device!!
    m_testbed->sync_device(*m_render_buffer, testbed_device);
    {
        m_render_buffer->reset_accumulation();
        m_render_buffer->clear_frame(stream);
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

        auto device_guard = use_device(stream, *m_render_buffer, device); // underlying device is the same as testbed_device.
        testbed_device.render_buffer_view().clear(testbed_device.stream());
	    vec2 focal_length = m_testbed->calc_focal_length(testbed_device.render_buffer_view().resolution, 
            m_testbed->m_relative_focal_length, m_testbed->m_fov_axis, m_testbed->m_zoom);
        vec2 screen_center = cam.render_screen_center(m_testbed->m_screen_center);
        int visualized_dimension = -1;
        auto n_elements = product(m_resolution);
        std::vector<vec3> sun_positions = {sun.pos};
        Triangle* vo_triangles = vo.has_value() ? vo.value().gpu_triangles() : nullptr;
        size_t vo_count = vo.has_value() ? vo.value().cpu_triangles().size() : 0;
        m_testbed->render_nerf(testbed_device.stream(), testbed_device, testbed_device.render_buffer_view(), 
            testbed_device.nerf_network(), testbed_device.data().density_grid_bitfield_ptr, 
            focal_length, cam_matrix, cam_matrix, vec4(vec3(0.0), 1.0), screen_center, {}, visualized_dimension);
        if (m_display_shadow) {
            m_testbed->render_nerf_with_shadow(testbed_device.stream(), testbed_device, testbed_device.render_buffer_view(), 
                testbed_device.nerf_network(), testbed_device.data().density_grid_bitfield_ptr, 
                focal_length, cam_matrix, cam_matrix, vec4(vec3(0.0), 1.0), screen_center, {}, visualized_dimension, 
                sun_positions, vo_triangles, vo_count);
        }
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
        m_testbed->render_frame_epilogue(stream, cam_matrix, m_last_camera, screen_center, 
            m_testbed->m_relative_focal_length, {}, {}, *m_render_buffer);

    }
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    m_last_camera = cam_matrix;
    m_last_sun = sun;
    m_last_vo = vo_matrix;
    m_is_dirty = false;
    return true;
}

void NerfWorld::imgui(float frame_time) {
    m_render_ms = frame_time;
	if (ImGui::Begin("Nerf Settings")) {
        if (ImGui::RadioButton("Toggle shadow On NeRF", m_display_shadow)) {
            m_display_shadow = !m_display_shadow;
            m_is_dirty = true;
        }
        ImGui::Text("FPS: %.3f", 1000.0 / frame_time);
        ImGui::SliderFloat("Target FPS: ", &m_dynamic_res_target_fps, 1, 25, "%.3f", 1.0f);
        ImGui::SliderInt("Fixed res factor: ", &m_fixed_res_factor, 8, 64);
    }
    ImGui::End();
}

__global__ void debug_depth(const uint32_t n_elements, vec4* __restrict__ rgba, float* __restrict__ depth) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
    if (i == n_elements * 4 / 7) {
        printf("NERF DEPTH: %.5f\n", depth[i]);
    }
}

}