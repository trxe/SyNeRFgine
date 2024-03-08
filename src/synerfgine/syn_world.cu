#include <synerfgine/cuda_helpers.h>
#include <synerfgine/material.h>
#include <synerfgine/syn_world.h>

#include <tiny-cuda-nn/common.h>
#include <iostream>
#include <filesystem>
#include <winnt.h>
#include <winuser.h>

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

namespace fs = std::filesystem;
using namespace tcnn;
using ngp::GLTexture;

// static bool is_first = true;

__device__ vec3 vec_to_col(const vec3& v) {
    return normalize(v) * 0.5f + vec3(0.5f);
}

__device__ vec3 reflect_ray(const vec3& incident, const vec3& normal);

__global__ void init_buffer(const uint32_t n_elements, vec3 col, float maxd, vec4* __restrict__ rgba, float* __restrict__ depth);

__global__ void gpu_draw_object(const uint32_t n_elements, const uint32_t width, const uint32_t height, const uint32_t tri_count,
    vec3* __restrict__ ray_origins, vec3* __restrict__ ray_directions, vec3* __restrict__ surface_normals, vec3* __restrict__ ray_scatters, 
    const Triangle* __restrict__ triangles, const vec3 sun_pos,
    vec4* __restrict__ rgba, float* __restrict__ depth, NerfPayload* __restrict__ payloads, const Material material);

__global__ void debug_draw_rays(const uint32_t n_elements, const uint32_t width, const uint32_t height, 
    // vec4* __restrict__ in_rgba, float* __restrict__ in_depth, 
    NerfPayload* __restrict__ in_pld, 
    vec3* __restrict__ ray_origins, vec3* __restrict__ ray_directions, vec3* __restrict__ ray_normals, 
    vec4* __restrict__ rgba, float* __restrict__ depth, ImgBuffers buffer_type);

__global__ void debug_triangle_vertices(const uint32_t n_elements, const Triangle* __restrict__ triangles,
    vec3* __restrict__ ray_origins, vec3* __restrict__ ray_directions);

__global__ void debug_pos(const uint32_t n_elements, vec3 pos, vec3 col, float radius, vec3* __restrict__ ray_origins, 
    vec3* __restrict__ ray_directions, vec4* __restrict__ rgba, float* __restrict__ depth);

__global__ void set_depth_buffer(const uint32_t n_elements, float* __restrict__ depth, float val);

__global__ void scatter_rays(const uint32_t n_elements, NerfPayload* __restrict__ payloads, 
    NerfPayload* __restrict__ prev_payloads, vec3* __restrict__ scatters);

__global__ void add_reflection_nerf(const uint32_t n_elements, mat4x3 camera_matrix,
	vec4* __restrict__ rgba, float* __restrict__ depth, // NerfPayload* __restrict__ payloads,
	vec4* __restrict__ frame_buffer, float* __restrict__ depth_buffer);

__global__ void init_rays_cam(
	uint32_t n_elements,
	uint32_t sample_index,
	vec3* __restrict__ positions,
	vec3* __restrict__ directions,
	ivec2 resolution,
	vec2 focal_length,
	mat4x3 camera_matrix0,
	mat4x3 camera_matrix1,
	vec4 rolling_shutter,
	vec2 screen_center,
	vec3 parallax_shift,
	bool snap_to_pixel_centers,
	// BoundingBox render_aabb,
	// mat3 render_aabb_to_local,
	float near_distance,
	float plane_z,
	float aperture_size,
	Foveation foveation,
	Lens lens,
	// Buffer2DView<const vec4> envmap,
	ngp::NerfPayload* __restrict__ payloads,
	// vec4* __restrict__ frame_buffer,
	float* __restrict__ depth_buffer,
	Buffer2DView<const uint8_t> hidden_area_mask,
	Buffer2DView<const vec2> distortion,
	ERenderMode render_mode
);

__global__ void debug_depth_syn(const uint32_t n_elements, vec4* __restrict__ rgba, float* __restrict__ depth);

SyntheticWorld::SyntheticWorld() {
	m_rgba_render_textures = std::make_shared<GLTexture>();
	m_depth_render_textures = std::make_shared<GLTexture>();

	m_render_buffer = std::make_shared<CudaRenderBuffer>(m_rgba_render_textures, m_depth_render_textures);
    m_render_buffer_view = m_render_buffer->view();
	m_render_buffer->disable_dlss();
    release();

    m_sun.pos = {1.027f, 3.389f, 3.803f};
}

SyntheticWorld::~SyntheticWorld() {
    release();
}

void SyntheticWorld::release() {
    m_nerf_payloads.check_guards();
    m_nerf_payloads.free_memory();
    m_nerf_payloads_refl.check_guards();
    m_nerf_payloads_refl.free_memory();
    m_gpu_positions.check_guards();
    m_gpu_positions.free_memory();
    m_gpu_directions.check_guards();
    m_gpu_directions.free_memory();
    m_gpu_normals.check_guards();
    m_gpu_normals.free_memory();
    m_gpu_scatters.check_guards();
    m_gpu_scatters.free_memory();
}

bool SyntheticWorld::handle_user_input(const ivec2& resolution) {
    is_buffer_outdated = m_camera.handle_user_input() | m_sun.handle_user_input(resolution);
    return is_buffer_outdated;
}

bool SyntheticWorld::handle(CudaDevice& device, const ivec2& resolution) {
    auto stream = device.stream();
    device.render_buffer_view().clear(stream);

    if (resolution != m_resolution) {
        m_resolution = resolution;
        m_rgba_render_textures->resize(resolution, 4);
        m_depth_render_textures->resize(resolution, 1);
        m_render_buffer->resize(resolution);
        m_render_buffer_view = m_render_buffer->view();
    }
    auto n_elements = m_resolution.x * m_resolution.y;

    auto& cam = m_camera;
    auto cam_matrix = cam.get_matrix();
    cam.set_resolution(m_resolution);
    
    auto device_guard = use_device(stream, *m_render_buffer, device);
    resize_gpu_buffers(n_elements);
    generate_rays_async(device);
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

    bool changed_depth = cam_matrix == m_last_camera;
    if (m_object.has_value()) {
        VirtualObject& vo = m_object.value();
        changed_depth = changed_depth & vo.update_triangles(stream);
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
        draw_object_async(device, vo);
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    }
    m_last_camera = cam_matrix;
    return true;
}

bool SyntheticWorld::shoot_network(CudaDevice& device, const ivec2& resolution, ngp::Testbed& testbed) {
    if (!m_display_shadow) {
        return false;
    }
    auto stream = device.stream();
    auto& testbed_device = testbed.primary_device();
    auto nerf_network = testbed_device.nerf_network();
    if (resolution != m_resolution) {
        m_resolution = resolution;
    }

    if (!m_object.has_value()) { 
        linear_kernel(debug_depth_syn, 0, stream, product(m_resolution), m_render_buffer_view.frame_buffer, m_render_buffer_view.depth_buffer);
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
        return false; 
    }

    uint32_t n_elements = m_resolution.x * m_resolution.y;
    {
        m_gpu_positions.check_guards();
        m_gpu_directions.check_guards();
        m_gpu_normals.check_guards();
        m_gpu_scatters.check_guards();
        m_nerf_payloads.check_guards();
        m_nerf_payloads_refl.check_guards();
        m_shadow_coeffs.check_guards();
        linear_kernel(scatter_rays, 0, stream, n_elements, m_nerf_payloads_refl.data(), m_nerf_payloads.data(), m_gpu_scatters.data());
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    }

    if (m_buffer_type != ImgBuffers::Final) {
        linear_kernel(debug_draw_rays, 0, stream, n_elements, m_resolution.x, m_resolution.y, 
            // rays_hit.rgba,
            // rays_hit.depth,
            m_display_nerf_payload_refl ? m_nerf_payloads_refl.data() : m_nerf_payloads.data(),
            m_gpu_positions.data(),
            m_gpu_directions.data(),
            m_gpu_normals.data(),
            m_render_buffer_view.frame_buffer,
            m_render_buffer_view.depth_buffer,
            m_buffer_type);
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
        return true;
    }

	ngp::Testbed::NerfTracer shadow_tracer;
    {
        vec3 sun_pos = m_camera.get_matrix() * vec4(m_sun.pos, 1.0);
        auto device_guard = use_device(stream, *m_render_buffer, device);
        vec3 center = m_object.has_value() ? m_object.value().get_center() : vec3(0.0);
        shadow_tracer.shoot_shadow_rays(m_nerf_payloads.data(), m_render_buffer_view, m_shadow_coeffs.data(),
            nerf_network, testbed_device.data().density_grid_bitfield_ptr, sun_pos, center,
            testbed.m_nerf.max_cascade, testbed.m_render_aabb, testbed.m_render_aabb_to_local, 
            m_filter_type, m_kernel_size, m_std_dev, stream);
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    }

	// ngp::Testbed::NerfTracer refl_tracer;
    // {
    //     auto device_guard = use_device(stream, *m_render_buffer, device);
    //     vec3 center = m_object.has_value() ? m_object.value().get_center() : vec3(0.0);
    //     auto cam_mat = m_camera.get_matrix();
    //     auto focal_len = m_camera.get_focal_length(m_resolution);
    //     float depth_scale = 1.0f / testbed.m_nerf.training.dataset.scale;
    //     refl_tracer.init_rays_from_payload(m_resolution, testbed.m_nerf_network, stream);
    //     CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    //     refl_tracer.trace(testbed.m_nerf_network, testbed.m_render_aabb, testbed.m_render_aabb_to_local, testbed.m_render_aabb,
    //         focal_len, testbed.m_nerf.cone_angle_constant, 
	// 		testbed_device.data().density_grid_bitfield_ptr,
	// 		testbed.m_render_mode,
    //         cam_mat,
    //         depth_scale,
	// 		testbed.m_visualized_layer,
	// 		testbed.m_visualized_dimension,
	// 		testbed.m_nerf.rgb_activation,
	// 		testbed.m_nerf.density_activation,
	// 		testbed.m_nerf.show_accel,
	// 		testbed.m_nerf.max_cascade,
	// 		testbed.m_nerf.render_min_transmittance,
	// 		testbed.m_nerf.glow_y_cutoff,
	// 		testbed.m_nerf.glow_mode,
    //         nullptr,
	// 		stream
    //     );
    //     CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    //     RaysNerfSoa& rays_hit = refl_tracer.rays_hit();
    //     linear_kernel(add_reflection_nerf, 0, stream, n_elements, 
    //         cam_mat,
    //         rays_hit.rgba,
    //         rays_hit.depth,
    //         m_render_buffer_view.frame_buffer,
    //         m_render_buffer_view.depth_buffer
    //     );
    //     CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    // }
    return true;
}

void SyntheticWorld::debug_visualize_pos(CudaDevice& device, const vec3& pos, const vec3& col, float sphere_size) {
    auto stream = device.stream();
    auto device_guard = use_device(stream, *m_render_buffer, device);
    auto n_elements = m_resolution.x * m_resolution.y;
    generate_rays_async(device);
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    linear_kernel(debug_pos, 0, stream, n_elements, 
        pos,
        col,
        sphere_size,
        m_gpu_positions.data(),
        m_gpu_directions.data(),
        m_render_buffer_view.frame_buffer,
        m_render_buffer_view.depth_buffer
    );
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
}

void SyntheticWorld::create_object(const std::string& filename) {
    delete_object();
    m_object.emplace(filename.c_str(), filename);
}

void SyntheticWorld::generate_rays_async(CudaDevice& device) {
	if (!is_buffer_outdated) {
		return;
	}
    cudaStream_t stream = device.stream();
    uint32_t n_elements = m_resolution.x * m_resolution.y;
	auto buf_size = sizeof(vec3) * n_elements;
    if (m_gpu_positions.size() != buf_size) {
    }

	vec2 focal_length = m_camera.get_focal_length(device.render_buffer_view().resolution);
	vec2 screen_center = m_camera.render_screen_center(camera_default::screen_center);

    auto& render_buf = device.render_buffer_view();

    linear_kernel(init_rays_cam, 0, stream, n_elements,
        render_buf.spp,
        m_gpu_positions.data(),
        m_gpu_directions.data(),
        m_resolution,
        focal_length,
		m_camera.get_matrix(),
		m_camera.get_matrix(),
		vec4(0.0f), // rolling_shutter
        screen_center,
		vec3(0.0),
		true,
		// render_aabb,
		// render_aabb_to_local,
		m_camera.m_ndc_znear,
		1.0f, // plane_z
		1.0f, // aperture_size
		ngp::Foveation{},
		ngp::Lens{}, // default perspective lens
		// envmap,
        m_nerf_payloads.data(),
		// render_buf.frame_buffer,
		render_buf.depth_buffer,
		Buffer2DView<const uint8_t>{}, // hidden_area_mask
		Buffer2DView<const vec2>{}, // distortion
		ERenderMode::Shade
    );
	is_buffer_outdated = false;
}


void SyntheticWorld::draw_object_async(CudaDevice& device, VirtualObject& vo) {
    auto stream = device.stream();
    auto n_elements = m_resolution.x * m_resolution.y;

    uint32_t tri_count = static_cast<uint32_t>(vo.cpu_triangles().size());
    {
        m_nerf_payloads.check_guards();
        m_nerf_payloads.resize(n_elements);
        vec3 sun_pos = m_camera.get_matrix() * vec4(m_sun.pos, 1.0);
        linear_kernel(gpu_draw_object, 0, stream, n_elements,
            m_resolution.x, 
            m_resolution.y, 
            tri_count,
            m_gpu_positions.data(),
            m_gpu_directions.data(),
            m_gpu_normals.data(),
            m_gpu_scatters.data(),
            vo.gpu_triangles(),
            sun_pos,
            m_render_buffer_view.frame_buffer, 
            m_render_buffer_view.depth_buffer,
            m_nerf_payloads.data(),
            m_object.value().get_material()
        );
    }
}

__global__ void gpu_draw_object(const uint32_t n_elements, const uint32_t width, const uint32_t height, const uint32_t tri_count,
    vec3* __restrict__ ray_origins, vec3* __restrict__ ray_directions, vec3* __restrict__ surface_normals, vec3* __restrict__ ray_scatters, 
    const Triangle* __restrict__ triangles, vec3 sun_pos,
    vec4* __restrict__ rgba, float* __restrict__ depth, NerfPayload* __restrict__ payloads, const Material material) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
    vec3 rd = ray_directions[i];
    vec3 ro = ray_origins[i];
    vec4 local_color;
    float dt = MAX_DEPTH();
    vec3 normal{};
    vec3 reflect_vec{};
    for (size_t k = 0; k < tri_count; ++k) {
        float t = triangles[k].ray_intersect(ro, rd);
        if (t < dt && t > ngp::MIN_RT_DIST) {
            dt = t;
            normal = triangles[k].normal();
        }
    }

    if (dt < ngp::MAX_RT_DIST) {
        ro += rd * dt;
        vec3 view_vec = normalize(-rd);
        vec3 to_sun = normalize(sun_pos - ro);
        float NdotL = dot(normal, to_sun);
        reflect_vec = reflect_ray(to_sun, normal);
        float RdotV = dot(reflect_vec, view_vec);
        vec3 primary_color = material.ka + max(0.0f, NdotL) * material.kd;
        vec3 secondary_color =  pow(max(0.0f, RdotV), material.n) * material.ks;
        local_color = vec4(min(vec3(1.0), primary_color + secondary_color), 1.0);
    }
    NerfPayload& payload = payloads[i];
    if (depth[i] > dt) {
        rgba[i] = local_color;
        depth[i] = dt;
        payload.max_weight = 0.0f;

        vec3 full_dir = sun_pos - ro;
        float n = length(full_dir);
        payload.origin = ro;
        payload.dir = normalize(full_dir);
        payload.t = n;
        payload.idx = i;
        payload.n_steps = 0;
        payload.alive = true;
    } else {
        payload.origin = vec3(0.0);
        payload.dir = vec3(0.0);
        payload.t = 0.0f;
        payload.idx = i;
        payload.n_steps = 0;
        payload.alive = false;
    }
    surface_normals[i] = normal;
    ray_scatters[i] = reflect_vec;
}

void SyntheticWorld::imgui(float frame_time) {
	static std::string imgui_error_string = "";

    m_camera.imgui();

	if (ImGui::CollapsingHeader("Toggle Buffer Views", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::RadioButton("Toggle view for Reflection NerfPayload", m_display_nerf_payload_refl)) {
            m_display_nerf_payload_refl = !m_display_nerf_payload_refl;
        }
        ImGui::Combo("Buffer Type", (int*)(&m_buffer_type), buffer_names, 
            sizeof(buffer_names) / sizeof(const char*));
    }
	if (ImGui::CollapsingHeader("Load Virtual Object", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::RadioButton("Toggle Shadow on Virtual Object", m_display_shadow)) {
            m_display_shadow = !m_display_shadow;
        }
        ImGui::SetNextItemOpen(m_show_kernel_settings);
        if (ImGui::TreeNode("Shadow Kernel")) {
            ImGui::Combo("Filter Type", (int*)(&m_filter_type), filter_names, 
                sizeof(filter_names) / sizeof(const char*));
            if (m_filter_type == ImgFilters::Box || m_filter_type == ImgFilters::Gaussian) {
                ImGui::InputInt("VO Shadow Kernel Size", &m_kernel_size);
            }
            if (m_filter_type == ImgFilters::Gaussian) {
                ImGui::SliderFloat("VO Std Dev", &m_std_dev, 0.0, 100.0f);
            }
            m_show_kernel_settings = true;
            ImGui::TreePop();
        } else {
            m_show_kernel_settings = false;
        }
        if (ImGui::TreeNode("Light Source")) {
            ImGui::Text("Control Virtual Light source");
            ImGui::SliderFloat3("Light position", m_sun.pos.data(), -5.0f, 5.0);
            ImGui::TreePop();
        }
		if (ImGui::TreeNode("Virtual Object")) {
            ImGui::Text("Add Virtual Object (.obj only)");
            ImGui::InputText("##PathFile", sng::virtual_object_fp, 1024);
            ImGui::SameLine();
            static std::string vo_path_load_error_string = "";
            if (ImGui::Button("Load")) {
                try {
                    create_object(sng::virtual_object_fp);
                } catch (const std::exception& e) {
                    ImGui::OpenPopup("Virtual object path load error");
                    vo_path_load_error_string = std::string{"Failed to load object path: "} + e.what();
                }
            }
            if (ImGui::BeginPopupModal("Virtual object path load error", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
                ImGui::Text("%s", vo_path_load_error_string.c_str());
                if (ImGui::Button("OK", ImVec2(120, 0))) {
                    ImGui::CloseCurrentPopup();
                }
                ImGui::EndPopup();
            }
            if (m_object.has_value()) {
                VirtualObject& obj = m_object.value();
                obj.imgui();
                if (ImGui::Button("Delete")) {
                    delete_object();
                }
            }
            ImGui::TreePop();
		}
	}
}

__device__ vec3 reflect_ray(const vec3& incident, const vec3& normal) {
    return dot(incident, normal) * 2.0f * normal - incident;
}

__global__ void debug_draw_rays(const uint32_t n_elements, const uint32_t width, const uint32_t height, 
    // vec4* __restrict__ in_rgba, float* __restrict__ in_depth, 
    NerfPayload* __restrict__ in_pld, 
    vec3* __restrict__ ray_origins, vec3* __restrict__ ray_directions, vec3* __restrict__ ray_normals, 
    vec4* __restrict__ rgba, float* __restrict__ depth, ImgBuffers buffer_type) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
    vec4 tmp_rgba = rgba[i];
    vec3 tmp_rgb;
    NerfPayload& pld = in_pld[i];
    switch (buffer_type) {
    // case ImgBuffers::ReflPayloadRGBA:
    //     tmp_rgba = in_rgba[i];
    //     break;
    case ImgBuffers::ReflPayloadDir:
        tmp_rgb = vec3(pld.alive ? pld.dir : 0.0);
        tmp_rgba = vec4(vec_to_col(tmp_rgb), 1.0);
        break;
    case ImgBuffers::ReflPayloadOrigin:
        tmp_rgb = vec3(pld.alive ? pld.origin : 0.0);
        tmp_rgba = vec4(vec_to_col(tmp_rgb), 1.0);
    // case ImgBuffers::ReflPayloadDepth:
    //     tmp_rgba = vec4(vec3(clamp(in_depth[i], 0.0f, 1.0f)), 1.0);
    //     break;
    case ImgBuffers::WorldOrigin:
        tmp_rgba = vec4(vec_to_col(ray_origins[i]), 1.0);
        break;
    case ImgBuffers::WorldDir:
        tmp_rgba = vec4(vec_to_col(ray_directions[i]), 1.0);
        break;
    case ImgBuffers::WorldNormal:
        tmp_rgba = vec4(vec_to_col(ray_normals[i]), 1.0);
        break;
    default:
        return;
    }
    rgba[i] = tmp_rgba;
    depth[i] = 0.0f;  // to put as top layer
}

__global__ void debug_triangle_vertices(const uint32_t n_elements, const Triangle* __restrict__ triangles, 
    vec3* __restrict__ ray_origins, vec3* __restrict__ ray_directions) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
    const Triangle* tri = &triangles[i];
    printf("%i: [%f %f %f], [%f %f %f], [%f %f %f]\n", i, 
        tri->a.r, tri->a.g, tri->a.b,
        tri->b.r, tri->b.g, tri->b.b,
        tri->c.r, tri->c.g, tri->c.b);
    // printf("%i: pos [%f %f %f], dir [%f %f %f]\n", i, 
    //     ray_origins[i].r, ray_origins[i].g, ray_origins[i].b, 
    //     ray_directions[i].r, ray_directions[i].g, ray_directions[i].b
    // );
}

__global__ void debug_pos(const uint32_t n_elements, vec3 pos, vec3 col, float radius, vec3* __restrict__ ray_origins, vec3* __restrict__ ray_directions, vec4* __restrict__ rgba, float* __restrict__ depth) {
    constexpr float TMIN = 0.000001;
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
    const float TMAX = depth[i];
    vec3 ro = ray_origins[i];
    vec3 rd = ray_directions[i];

    vec3 l  = pos - ro;
    float a = dot(rd, rd);
    float b = 2 * dot(rd, l);
    float c = dot(l,l) - radius * radius;
    float det  = b * b - 4 * a * c;
    if (det < 0) return;
    float t = (-b - sqrt(det)) / 2;
    if (t < TMIN) {
        t = (-b + sqrt(det)) / 2;
    }
    if (t >= TMIN && t < TMAX) {
        float ratio = t * t / length2(ro - pos);
        rgba[i] = vec4(col * ratio, 1.0f);
        depth[i] = t;
    }
}

__global__ void init_buffer(const uint32_t n_elements, vec3 col, float maxd, vec4* __restrict__ rgba, float* __restrict__ depth) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
    rgba[i] = vec4(col, 1.0);
    depth[i] = maxd;
}

__global__ void set_depth_buffer(const uint32_t n_elements, float* __restrict__ depth, float val) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
    depth[i] = val;
}

__global__ void scatter_rays(const uint32_t n_elements, NerfPayload* __restrict__ payloads, 
    NerfPayload* __restrict__ prev_payloads, vec3* __restrict__ scatters) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
    payloads[i] = prev_payloads[i];
    payloads[i].dir = scatters[i];
}

__global__ void add_reflection_nerf(
	const uint32_t n_elements,
	mat4x3 camera_matrix,
	vec4* __restrict__ rgba,
	float* __restrict__ depth,
	// NerfPayload* __restrict__ payloads,
	vec4* __restrict__ frame_buffer,
	float* __restrict__ depth_buffer
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
    // NerfPayload& payload = payloads[i];
    vec4 tmp = rgba[i];
    vec4 fb = frame_buffer[i];
    tmp.rgb() = srgb_to_linear(tmp.rgb());
    frame_buffer[i] = fb * 0.5f + tmp * 0.5f;
    // rgba[i] = fb * 0.5f + tmp * 0.5f;
}

__global__ void init_rays_cam(
	uint32_t n_elements,
	uint32_t sample_index,
	vec3* __restrict__ positions,
	vec3* __restrict__ directions,
	ivec2 resolution,
	vec2 focal_length,
	mat4x3 camera_matrix0,
	mat4x3 camera_matrix1,
	vec4 rolling_shutter,
	vec2 screen_center,
	vec3 parallax_shift,
	bool snap_to_pixel_centers,
	// BoundingBox render_aabb,
	// mat3 render_aabb_to_local,
	float near_distance,
	float plane_z,
	float aperture_size,
	Foveation foveation,
	Lens lens,
	// Buffer2DView<const vec4> envmap,
	ngp::NerfPayload* __restrict__ payloads,
	// vec4* __restrict__ frame_buffer,
	float* __restrict__ depth_buffer,
	Buffer2DView<const uint8_t> hidden_area_mask,
	Buffer2DView<const vec2> distortion,
	ERenderMode render_mode
) {
	uint32_t idx =threadIdx.x + blockDim.x * blockIdx.x; 
	uint32_t x = idx % resolution.x;
	uint32_t y = idx / resolution.x;

	if (idx > n_elements) {
		return;
	}

	if (plane_z < 0) {
		aperture_size = 0.0;
	}

	vec2 pixel_offset = ld_random_pixel_offset(snap_to_pixel_centers ? 0 : sample_index);
	vec2 uv = vec2{(float)x + pixel_offset.x, (float)y + pixel_offset.y} / vec2(resolution);
	mat4x3 camera = get_xform_given_rolling_shutter({camera_matrix0, camera_matrix1}, rolling_shutter, uv, ld_random_val(sample_index, idx * 72239731));
	ngp::NerfPayload& payload = payloads[idx];

	// returns ray in world space
	Ray ray = uv_to_ray(
		sample_index,
		uv,
		resolution,
		focal_length,
		camera,
		screen_center
	);

	depth_buffer[idx] = MAX_DEPTH();

	if (!ray.is_valid()) {
        positions[idx] = ray.o;
        directions[idx] = vec3(0.0);
		payload.origin = ray.o;
		payload.alive = false;
		return;
	}

	float n = length(ray.d);
	positions[idx] = ray.o;
	directions[idx] = (1.0f/n) * ray.d;
	payload.origin = ray.o;
	payload.dir = (1.0f/n) * ray.d;
	payload.t = -plane_z*n;
	payload.idx = idx;
	payload.n_steps = 0;
	payload.alive = ray.is_valid();

	ray.d = normalize(ray.d);
    positions[idx] = ray.o;
    directions[idx] = ray.d;
}

__global__ void debug_depth_syn(const uint32_t n_elements, vec4* __restrict__ rgba, float* __restrict__ depth) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
    if (i % 10000 == 0) {
        printf("SYN DEPTH[%d]: %.5f\n", i, depth[i]);
    }
}

}