#include <synerfgine/cuda_helpers.h>
#include <synerfgine/material.h>
#include <synerfgine/syn_world.h>

#include <tiny-cuda-nn/common.h>
#include <iostream>
#include <filesystem>

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

__global__ void init_buffer(const uint32_t n_elements, vec4* __restrict__ rgba, float* __restrict__ depth);

__global__ void debug_paint(const uint32_t n_elements, const uint32_t width, const uint32_t height, 
    vec4* __restrict__ rgba, float* __restrict__ depth);

__global__ void gpu_draw_object(const uint32_t n_elements, const uint32_t width, const uint32_t height, const uint32_t tri_count,
    vec3* __restrict__ ray_origins, vec3* __restrict__ ray_directions, const Triangle* __restrict__ triangles, const Light sun,
    vec4* __restrict__ rgba, float* __restrict__ depth, NerfPayload* __restrict__ payloads, const Material material);

__global__ void debug_draw_rays(const uint32_t n_elements, const uint32_t width, const uint32_t height, 
    vec3* __restrict__ ray_origins, vec3* __restrict__ ray_directions, 
    vec4* __restrict__ rgba, float* __restrict__ depth);

__global__ void debug_triangle_vertices(const uint32_t n_elements, const Triangle* __restrict__ triangles,
    vec3* __restrict__ ray_origins, vec3* __restrict__ ray_directions);

__global__ void debug_pos(const uint32_t n_elements, vec3 pos, vec3 col, float radius, vec3* __restrict__ ray_origins, 
    vec3* __restrict__ ray_directions, vec4* __restrict__ rgba, float* __restrict__ depth);

__global__ void set_depth_buffer(const uint32_t n_elements, float* __restrict__ depth, float val);

SyntheticWorld::SyntheticWorld() {
	m_rgba_render_textures = std::make_shared<GLTexture>();
	m_depth_render_textures = std::make_shared<GLTexture>();

	m_render_buffer = std::make_shared<CudaRenderBuffer>(m_rgba_render_textures, m_depth_render_textures);
    m_render_buffer_view = m_render_buffer->view();
	m_render_buffer->disable_dlss();
    m_nerf_payloads.check_guards();
    m_nerf_payloads.free_memory();
    m_nerf_payloads.resize(0);
}

bool SyntheticWorld::handle_user_input(const ivec2& resolution) {
    return m_camera.handle_user_input() | m_sun.handle_user_input(resolution);
}

bool SyntheticWorld::handle(CudaDevice& device, const ivec2& resolution) {
    auto stream = device.stream();
    device.render_buffer_view().clear(stream);

    auto n_elements = m_resolution.x * m_resolution.y;
    if (resolution != m_resolution) {
        m_resolution = resolution;
        m_rgba_render_textures->resize(resolution, 4);
        m_depth_render_textures->resize(resolution, 1);
        m_render_buffer->resize(resolution);
        m_render_buffer_view = m_render_buffer->view();
    }

    auto& cam = m_camera;
    auto cam_matrix = cam.get_matrix();
    cam.set_resolution(m_resolution);
    
    auto device_guard = use_device(stream, *m_render_buffer, device);
    cam.generate_rays_async(device);
    bool changed_depth = cam_matrix == m_last_camera;
    linear_kernel(init_buffer, 0, stream, n_elements, m_render_buffer_view.frame_buffer, m_render_buffer_view.depth_buffer);
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

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
    {
        auto n_elements = resolution.x * resolution.y;
        m_nerf_payloads.check_guards();
        m_nerf_payloads.resize(n_elements);
        m_shadow_coeffs.check_guards();
        m_shadow_coeffs.resize(n_elements);
    }

	ngp::Testbed::NerfTracer tracer;
    {
        auto device_guard = use_device(stream, *m_render_buffer, device);
        vec3 center = m_object.has_value() ? m_object.value().get_center() : vec3(0.0);
        tracer.shoot_shadow_rays(m_nerf_payloads.data(), m_render_buffer_view, m_shadow_coeffs.data(),
            nerf_network, testbed_device.data().density_grid_bitfield_ptr, sun_pos(), center,
            testbed.m_nerf.max_cascade, testbed.m_render_aabb, testbed.m_render_aabb_to_local, stream);
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    }
    return true;
}

bool SyntheticWorld::debug_visualize_pos(CudaDevice& device, const vec3& pos, const vec3& col, float sphere_size) {
    auto stream = device.stream();
    auto device_guard = use_device(stream, *m_render_buffer, device);
    auto& cam = m_camera;
    cam.generate_rays_async(device);
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    auto n_elements = m_resolution.x * m_resolution.y;
    linear_kernel(debug_pos, 0, stream, n_elements, 
        pos,
        col,
        sphere_size,
        cam.gpu_positions(),
        cam.gpu_directions(),
        m_render_buffer_view.frame_buffer,
        m_render_buffer_view.depth_buffer
    );
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
}

void SyntheticWorld::create_object(const std::string& filename) {
    delete_object();
    m_object.emplace(filename.c_str(), filename);
}

void SyntheticWorld::draw_object_async(CudaDevice& device, VirtualObject& virtual_object) {
    auto& cam = m_camera;
    auto stream = device.stream();
    auto n_elements = m_resolution.x * m_resolution.y;
    uint32_t tri_count = static_cast<uint32_t>(virtual_object.cpu_triangles().size());
    {
        m_nerf_payloads.check_guards();
        m_nerf_payloads.resize(n_elements);
        linear_kernel(gpu_draw_object, 0, stream, n_elements,
            m_resolution.x, 
            m_resolution.y, 
            tri_count,
            cam.gpu_positions(),
            cam.gpu_directions(),
            virtual_object.gpu_triangles(),
            m_sun,
            m_render_buffer_view.frame_buffer, 
            m_render_buffer_view.depth_buffer,
            m_nerf_payloads.data(),
            virtual_object.get_material()
            );
    }
}

__global__ void gpu_draw_object(const uint32_t n_elements, const uint32_t width, const uint32_t height, const uint32_t tri_count,
    vec3* __restrict__ ray_origins, vec3* __restrict__ ray_directions, const Triangle* __restrict__ triangles, const Light sun,
    vec4* __restrict__ rgba, float* __restrict__ depth, NerfPayload* __restrict__ payloads, const Material material) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
    vec3 rd = ray_directions[i];
    vec3 ro = ray_origins[i];
    vec4 local_color;
    float dt = depth[i];
    vec3 normal;
    for (size_t k = 0; k < tri_count; ++k) {
        float t = triangles[k].ray_intersect(ro, rd);
        if (t < dt && t > ngp::MIN_RT_DIST) {
            dt = t;
            normal = triangles[k].normal();
        }
    }

    if (dt < ngp::MAX_RT_DIST) {
        ro += rd * dt;
        vec3 view_vec = normalize(ray_origins[i] - ro);
        vec3 to_sun = normalize(sun.pos - ro);
        float NdotL = dot(normal, to_sun);
        vec3 reflect_vec = normal * 2.0f * NdotL - to_sun;
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

        vec3 pos = ro;
        vec3 full_dir = sun.pos - pos;
        float n = length(full_dir);
        payload.origin = pos;
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
}

void SyntheticWorld::imgui(float frame_time) {
	static std::string imgui_error_string = "";

	if (ImGui::Begin("Load Virtual Object")) {
        if (ImGui::RadioButton("Toggle Shadow on Virtual Object", m_display_shadow)) {
            m_display_shadow = !m_display_shadow;
        }

		ImGui::Text("Control Virtual Light source");
        ImGui::SliderFloat3("Light position", m_sun.pos.data(), -5.0f, 5.0);
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
		if (m_object.has_value() && ImGui::CollapsingHeader("Virtual Object", ImGuiTreeNodeFlags_DefaultOpen)) {
            VirtualObject& obj = m_object.value();
            obj.imgui();
            if (ImGui::Button("Delete")) {
				delete_object();
			}
		}
	}
	ImGui::End();
	if (ImGui::Begin("Camera")) {
		auto rd = camera().view_pos();
		ImGui::Text("View Pos: %f, %f, %f", rd.r, rd.g, rd.b);
		rd = camera().view_dir();
		ImGui::Text("View Dir: %f, %f, %f", rd.r, rd.g, rd.b);
		rd = camera().look_at();
		ImGui::Text("Look At: %f, %f, %f", rd.r, rd.g, rd.b);
		rd = m_sun.pos;
		ImGui::Text("Sun Pos: %f, %f, %f", rd.r, rd.g, rd.b);
		float fps = !frame_time ? std::numeric_limits<float>::max() : (1000.0f / frame_time);
		ImGui::Text("Frame: %.2f ms (%.1f FPS)", frame_time, fps);
		if (ImGui::Button("Reset Camera")) {
			mut_camera().reset_camera();
		}
	}
	ImGui::End();
}

__global__ void debug_draw_rays(const uint32_t n_elements, const uint32_t width, const uint32_t height, 
    vec3* __restrict__ ray_origins, vec3* __restrict__ ray_directions, 
    vec4* __restrict__ rgba, float* __restrict__ depth) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	// if (i == 0) { printf("DIR: %f %f %f\n", ray_directions[i].x, ray_directions[i].y, ray_directions[i].z); }
    rgba[i] = vec4(abs(ray_directions[i]), 1.0);
	// if (i % 100000 == 0) { printf("COL %i: %f %f %f %f\n", i, rgba[i].x, rgba[i].y, rgba[i].z, rgba[i].w); }
}

__global__ void debug_paint(const uint32_t n_elements, const uint32_t width, const uint32_t height, 
    vec4* __restrict__ rgba, float* __restrict__ depth) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
    float x = (float)(i % width) / (float)width;
    float y = (float)(i / height) / (float)height;
    rgba[i] = vec4(x, y, 0.0, 1.0);
    depth[i] = 0.5f;
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
    constexpr float TMAX = 100000.0;
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
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
    if (t >= TMIN || t < TMAX) {
        float ratio = t * t / length2(ro - pos);
        rgba[i] = vec4(col * ratio, 1.0f);
        depth[i] = t;
    }
}

__global__ void init_buffer(const uint32_t n_elements, vec4* __restrict__ rgba, float* __restrict__ depth) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
    rgba[i] = vec4(vec3(0.0), 1.0);
    depth[i] = ngp::MAX_RT_DIST;
}

__global__ void set_depth_buffer(const uint32_t n_elements, float* __restrict__ depth, float val) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
    depth[i] = val;
}

}