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
__global__ void debug_reset_depth(const uint32_t n_elements, float* __restrict__ depth);
__global__ void debug_init_rays_payloads(const uint32_t n_elements, NerfPayload* payloads,
    vec4* __restrict__ rgba, float* __restrict__ depth, NerfImgBuffers buf_type);

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
        vec4 rolling_shutter = vec4(vec3(0.0), 1.0);
        int visualized_dimension = -1;
        auto n_elements = product(m_resolution);
        std::vector<vec3> sun_positions = {sun.pos};
        Triangle* vo_triangles = vo.has_value() ? vo.value().gpu_triangles() : nullptr;
        size_t vo_count = vo.has_value() ? vo.value().cpu_triangles().size() : 0;
        m_testbed->render_nerf(testbed_device.stream(), testbed_device, testbed_device.render_buffer_view(), 
            testbed_device.nerf_network(), testbed_device.data().density_grid_bitfield_ptr, 
            focal_length, cam_matrix, cam_matrix, rolling_shutter, screen_center, {}, visualized_dimension);
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
	if (ImGui::CollapsingHeader("Nerf Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::RadioButton("Toggle shadow On NeRF", m_display_shadow)) {
            m_display_shadow = !m_display_shadow;
            m_is_dirty = true;
        }
        if (ImGui::TreeNode("Info")) {
            ImGui::Text("FPS: %.3f", 1000.0 / frame_time);
            ImGui::SliderFloat("Target FPS: ", &m_dynamic_res_target_fps, 1, 25, "%.3f", 1.0f);
            ImGui::SliderInt("Fixed res factor: ", &m_fixed_res_factor, 8, 64);
            ImGui::TreePop();
        }
    }
}

void NerfWorld::reset_frame(CudaDevice& device, const ivec2& resolution) {
    if (!m_testbed) return;
    constexpr float pixel_ratio = 1.0f;
    float factor = std::sqrt(pixel_ratio / m_render_ms * 1000.0f / m_dynamic_res_target_fps);
    factor = 8.f / (float)m_fixed_res_factor;
    factor = clamp(factor, 1.0f / 16.0f, 1.0f);

    auto stream = device.stream();
    auto new_resolution = downscale_resolution(resolution, factor);
    if (new_resolution != m_resolution) {
        m_testbed->m_nerf.training.dataset.scale = 1.0;
        m_resolution = new_resolution;
        m_rgba_render_textures->resize(m_resolution, 4);
        m_depth_render_textures->resize(m_resolution, 1);
        m_render_buffer->clear_frame(stream);
        m_render_buffer->resize(m_resolution);
        m_render_buffer->set_hidden_area_mask(nullptr);
        m_render_buffer->disable_dlss();
        m_render_buffer_view = m_render_buffer->view();
        auto n_elements = m_resolution.x * m_resolution.y;
        linear_kernel(debug_reset_depth, 0, stream, n_elements, m_render_buffer_view.depth_buffer);
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    }
}

bool NerfWorld::debug_init_rays(CudaDevice& device, const ivec2& resolution, const Camera& cam) {
	if (ImGui::CollapsingHeader("Toggle Buffer Views", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Combo("Buffer Type", (int*)(&m_buffer_type), nerf_buffer_names, 
            sizeof(nerf_buffer_names) / sizeof(const char*));
    }
    auto cam_matrix = cam.get_matrix();
    if (m_last_camera == cam_matrix) return false;

    if (!m_testbed) return false;
    reset_frame(device, resolution);
    auto stream = device.stream();
    auto& testbed_device = m_testbed->primary_device();
    testbed_device.set_render_buffer_view(m_render_buffer_view);
    // MUST use m_testbed's sync_device!!
    m_testbed->sync_device(*m_render_buffer, testbed_device);
    m_render_buffer->reset_accumulation();
    m_render_buffer->clear_frame(stream);
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

    {
        auto device_guard = use_device(stream, *m_render_buffer, device); // underlying device is the same as testbed_device.
        auto render_buffer = testbed_device.render_buffer_view();
        auto nerf_network = m_testbed->m_nerf_network;
        render_buffer.clear(testbed_device.stream());

	    vec2 focal_length = m_testbed->calc_focal_length(testbed_device.render_buffer_view().resolution, 
            m_testbed->m_relative_focal_length, m_testbed->m_fov_axis, m_testbed->m_zoom);
        vec2 screen_center = cam.render_screen_center(m_testbed->m_screen_center);
        vec4 rolling_shutter = vec4(vec3(0.0), 1.0);
        int visualized_dimension = -1;
        ngp::Testbed::NerfTracer tracer;
        tracer.init_rays_from_camera(
            render_buffer.spp,
            nerf_network,
            render_buffer.resolution,
            focal_length,
            cam.get_matrix(),
            rolling_shutter,
            screen_center,
            m_testbed->m_parallax_shift,
            m_testbed->m_snap_to_pixel_centers,
           m_testbed->m_render_aabb,
           m_testbed->m_render_aabb_to_local,
           m_testbed->m_render_near_distance,
            m_testbed->m_slice_plane_z + cam.scale(),
           m_testbed->m_aperture_size,
           Foveation{},
            Lens{},
            m_testbed->m_envmap.inference_view(),
            Buffer2DView<const vec2>{},
            render_buffer.frame_buffer,
            render_buffer.depth_buffer,
            render_buffer.hidden_area_mask ? render_buffer.hidden_area_mask->const_view() : Buffer2DView<const uint8_t>{},
            device.data().density_grid_bitfield_ptr,
           m_testbed->m_nerf.show_accel,
           m_testbed->m_nerf.max_cascade,
           m_testbed->m_nerf.cone_angle_constant,
           ERenderMode::Shade,
            stream
        );
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
        auto n_elements = m_resolution.x * m_resolution.y;
        linear_kernel(debug_init_rays_payloads, 0, stream, n_elements, 
            tracer.rays_hit().payload, render_buffer.frame_buffer, render_buffer.depth_buffer, m_buffer_type);
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    }
}

__global__ void debug_depth(const uint32_t n_elements, vec4* __restrict__ rgba, float* __restrict__ depth) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
    if (i == n_elements * 4 / 7) {
        printf("NERF DEPTH: %.5f\n", depth[i]);
    }
}

__global__ void debug_init_rays_payloads(const uint32_t n_elements, NerfPayload* payloads,
    vec4* __restrict__ rgba, float* __restrict__ depth, NerfImgBuffers buf_type) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
    if (buf_type == NerfImgBuffers::NerfWorldDir) {
        rgba[i] = normalize(payloads[i].dir) * 0.5f + vec3(0.5f);
    } else {
        rgba[i] = normalize(payloads[i].origin) * 0.5f + vec3(0.5f);
    }
    depth[i] = -0.1f;
}

__global__ void debug_reset_depth(const uint32_t n_elements, float* __restrict__ depth) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
    depth[i] = 100000.f;
}

}