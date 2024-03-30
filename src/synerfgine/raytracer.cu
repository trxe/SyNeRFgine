#include <synerfgine/raytracer.cuh>
#include <neural-graphics-primitives/nerf_device.cuh>
#include <neural-graphics-primitives/render_buffer.h>

namespace sng {
__device__ vec4 shade_object(const vec3& wi, SampledRay& ray, const uint32_t& shadow_count, HitRecord& hit_info,
	const Light* __restrict__ lights, const size_t& light_count, 
	const ObjectTransform* __restrict__ objects, const size_t& object_count, 
	const Material* __restrict__ materials, const size_t& material_count,
	const uint32_t& n_steps, const float& cone_angle_constant, 
	const uint8_t* __restrict__ density_grid, const uint32_t& min_mip, const uint32_t& max_mip, 
	const BoundingBox& render_aabb, const mat3& render_aabb_to_local,
	curandState_t& rand_state
) {
	if (hit_info.material_idx < 0) return vec4(0.0);
	const Material& material = materials[hit_info.material_idx];
	vec4 color{vec3(0.0), 1.0};
	vec3 L {0.0};
	for (size_t l = 0; l < light_count; ++l) {
		const Light& light = lights[l];
		for (size_t s = 0; s < shadow_count; ++s) {
			// vec3 lpos = light.sample(rand_state);
			vec3 lpos = light.sample();

			L = lpos - hit_info.pos;
			float full_dist = length(L);
			L = normalize(L);
			vec3 invL = vec3(1.0f) / L;
			int32_t obj_hit = -1; 
			float syn_shadow = depth_test_world(hit_info.pos, L, objects, object_count, hit_info.object_idx, obj_hit);
			float nerf_shadow = depth_test_nerf(syn_shadow + 1.0, n_steps, cone_angle_constant, hit_info.pos, L, invL, density_grid, min_mip, max_mip, render_aabb, render_aabb_to_local);
			float shadow_mask = smoothstep(min(min(nerf_shadow, syn_shadow), full_dist) / full_dist);
			vec3 R = reflect(L, hit_info.normal);
			vec3 V = normalize(-wi);
			color.rgb() += material.local_color(L, hit_info.normal, R, V, light) * shadow_mask;
		}
	}
	color.rgb() /= (float) shadow_count;
	color.rgb() += material.ka;
	material.scatter(hit_info, wi, ray, rand_state);
	return color;
}

__global__ void init_rays_with_payload_kernel_nerf(
	uint32_t sample_index,
	ivec2 resolution,
	vec2 focal_length,
	mat4x3 camera,
	vec2 screen_center,
	vec4* __restrict__ frame_buffer,
	float* __restrict__ depth_buffer,
	vec3* __restrict__ origin,
	vec3* __restrict__ dir,
	vec4* __restrict__ acc_rgba,
	float* __restrict__ acc_depth
) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= resolution.x || y >= resolution.y) {
		return;
	}

	uint32_t idx = x + resolution.x * y;

	vec2 uv = {(float)x / (float)resolution.x, (float)y / (float)resolution.y};
	Ray ray = uv_to_ray(
		sample_index,
		uv,
		resolution,
		focal_length,
		camera,
		screen_center
	);

	frame_buffer[idx] = vec4(vec3(0.0), 1.0);
	depth_buffer[idx] = MAX_DEPTH();
	acc_rgba[idx] = vec4(vec3(0.0), 1.0);
	acc_depth[idx] = MAX_DEPTH();

	ray.d = normalize(ray.d);
	origin[idx] = ray.o;
	dir[idx] = ray.d;
}

__global__ void raytrace(uint32_t n_elements, 
	const vec3* __restrict__ src_positions, 
	const vec3* __restrict__ src_directions, 
	mat4x3 camera,
	const Light* __restrict__ lights,
	size_t light_count,
	const Material* __restrict__ materials,
	size_t mat_count,
	const ObjectTransform* __restrict__ world,
	size_t world_count,
	float cone_angle_constant,
	uint32_t n_steps,
	bool show_nerf_shadow, 
	BoundingBox render_aabb,
	mat3 render_aabb_to_local,
	ivec2 focal_length,
	const uint8_t* __restrict__ density_grid,
	uint32_t min_mip,
	uint32_t max_mip,
	ImgBufferType buffer_type,
	size_t sample_count,
	size_t bounce_count,
	size_t shadow_count,
	float attenuation_coeff,
	float depth_offset,
	curandState_t* __restrict__ rand_state,
	vec4* __restrict__ acc_rgba, 
	float* __restrict__ acc_depth
) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) return;
	const vec3 up_vec = camera[0];

	vec3 shade{0.0f};
	vec3 normal{0.0f};
	vec3 view_pos{0.0f};
	vec3 next_pos{0.0f};
	vec3 view_dir{0.0f};
	vec3 next_dir{0.0f};
	// float depth{0.0f};
	for (size_t spp = 0; spp < sample_count; ++spp) {
		auto src_pos = vec3(0.0);
		auto src_dir = vec3(0.0);
		SampledRay ray;
		auto longi = curand_uniform(&rand_state[i]) * cone_angle_constant * tcnn::PI;
		auto latid = curand_uniform(&rand_state[i]) * cone_angle_constant * tcnn::PI;
		ray.pos = src_positions[i];
		ray.dir = cone_random(src_directions[i], up_vec, longi, latid);
		ray.pdf = 1.0f / (float)bounce_count;
		ray.attenuation = 1.0f;
		vec3 shade_s{0.0f};
		for (size_t bounce = 0; bounce < bounce_count; ++bounce) {
			src_pos = ray.pos;
			src_dir = ray.dir;
			HitRecord hit_info;
			int32_t hit_obj_id = -1;
			float d = depth_test_world(src_pos, src_dir, world, world_count, hit_obj_id, hit_info) + depth_offset;
			if (!bounce) {
				// depth += d;
				normal += hit_info.normal;
				view_pos += src_pos;
				view_dir += src_dir;
				next_pos += hit_info.pos;
			}
			if (hit_obj_id < 0) break;
			SampledRay next_ray;
			vec4 color = shade_object(src_dir, next_ray, shadow_count, hit_info, lights, light_count, world, world_count, materials, mat_count,
				n_steps, cone_angle_constant, density_grid, min_mip, max_mip, render_aabb, render_aabb_to_local, rand_state[i]);
			shade_s += color.rgb() * ray.pdf * ray.attenuation;
			if (!bounce) {
				next_dir += next_ray.dir;
			}
			ray = next_ray;
		}
		shade += shade_s;
	}
	float weight = (float)sample_count;
	view_pos /= weight;
	view_dir /= weight;
	next_pos /= weight;
	next_dir /= weight;
	normal /= weight;
	shade /= weight;
	float depth = dot(src_directions[i], next_pos - src_positions[i]);
	acc_depth[i] = depth;
	vec3 curr_shade = acc_rgba[i].rgb();
	switch (buffer_type) {
	case ImgBufferType::Normal:
		// acc_rgba[i].rgb() = vec3_to_col(normal);
		acc_rgba[i].rgb() = normal;
		break;
	case ImgBufferType::NextDirection:
		// acc_rgba[i].rgb() = vec3_to_col(next_dir);
		acc_rgba[i].rgb() = next_dir;
		break;
	case ImgBufferType::SrcDirection:
		// acc_rgba[i].rgb() = vec3_to_col(src_dir);
		acc_rgba[i].rgb() = view_dir;
		break;
	case ImgBufferType::NextOrigin:
		acc_rgba[i].rgb() = vec3_to_col(next_pos);
		break;
	case ImgBufferType::SrcOrigin:
		acc_rgba[i].rgb() = vec3_to_col(view_pos);
		break;
	case ImgBufferType::Depth:
		acc_rgba[i].rgb() = vec3(1.0 - fractf(depth / 10.0));
		break;
	default:
		if (dot(curr_shade, curr_shade) > 0.001f) {
			shade = shade * 0.5f + acc_rgba[i].rgb() * 0.5f;
		}
		acc_rgba[i].rgb() = shade;
		break;
	}
}

__global__ void overlay_nerf(ivec2 syn_res, 
	int syn_px_scale, 
	vec4* __restrict__ final_rgba, 
	float* __restrict__ final_depth, 
	vec4* __restrict__ syn_rgba, 
	float* __restrict__ syn_depth, 
	vec4* __restrict__ nerf_rgba, 
	float* __restrict__ nerf_depth,
	EColorSpace color_space,
	ETonemapCurve tonemap_curve,
	float exposure
) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= syn_res.x || y >= syn_res.y) {
		return;
	}

	ivec2 nerf_res = syn_res / syn_px_scale;
	ivec2 sc = {x, y};
	ivec2 nc = sc / syn_px_scale;

	vec4 srgba = syn_rgba[sc.x + sc.y * syn_res.x];
	float sdepth = syn_depth[sc.x + sc.y * syn_res.x];
	vec4 nrgba = nerf_rgba[nc.x + nc.y * nerf_res.x];
	float ndepth = nerf_depth[nc.x + nc.y * nerf_res.x];

	// DEBUG
	// if (sdepth < ndepth) return;
	// final_rgba[sc.x + sc.y * syn_res.x] = sdepth < ndepth ? srgba : nrgba;
	// final_depth[sc.x + sc.y * syn_res.x] = sdepth < ndepth ? sdepth : ndepth;

	// 2. applying exposure in linear space
	// srgba.rgb() *= pow(vec3(2.0f), exposure);
	srgba.rgb() = sng_tonemap(srgba.rgb(), tonemap_curve);
	final_rgba[sc.x + sc.y * syn_res.x] = color_space == EColorSpace::SRGB ? vec4(linear_to_srgb(srgba.rgb()), srgba.a) : srgba;
	final_depth[sc.x + sc.y * syn_res.x] = sdepth;
}

void RayTracer::enlarge(const ivec2& res) {
	if (m_render_buffer.out_resolution() == res) return;
    m_render_buffer.resize(res);
    size_t n_elements = product(res);
	n_elements = next_multiple(n_elements, size_t(BATCH_SIZE_GRANULARITY));
	auto scratch = allocate_workspace_and_distribute<
		vec3, // origin
		vec3, // dir
		vec4, // rgba
		float, // depth
		curandState_t // rand values
	>(
		m_stream_ray, &m_scratch_alloc,
		n_elements, n_elements, n_elements, n_elements, n_elements);

	m_rays[0].set(std::get<0>(scratch), std::get<1>(scratch), std::get<2>(scratch), std::get<3>(scratch), n_elements);
	m_rand_state = std::get<4>(scratch);

	linear_kernel(init_rand_state, 0, m_stream_ray, n_elements, m_rand_state);
	sync();
}

void RayTracer::init_rays_from_camera(
	uint32_t sample_index,
	const vec2& focal_length,
	const mat4x3& camera,
	const vec2& screen_center,
	bool snap_to_pixel_centers
) {
	// Make sure we have enough memory reserved to render at the requested resolution
	ivec2 res = resolution();
	enlarge(res);

	const dim3 threads = { 16, 8, 1 };
	const dim3 blocks = { div_round_up((uint32_t)res.x, threads.x), div_round_up((uint32_t)res.y, threads.y), 1 };
	init_rays_with_payload_kernel_nerf<<<blocks, threads, 0, m_stream_ray>>>(
		sample_index,
		res,
		focal_length,
		camera,
		screen_center,
		m_render_buffer.frame_buffer(),
		m_render_buffer.depth_buffer(),
		m_rays[0].origin,
		m_rays[0].dir,
		m_rays[0].rgba,
		m_rays[0].depth
	);
	sync();
}

void RayTracer::render(
	std::vector<VirtualObject>& h_vo, 
	std::vector<LightProbe>& light_probes, 
	const GPUMemory<Material>& materials, 
	const GPUMemory<Light>& lights, 
	const Testbed::View& view, 
	const vec2& screen_center,
	uint32_t sample_index,
	const vec2& focal_length,
	const float& depth_offset,
	bool snap_to_pixel_centers,
	const uint8_t* density_grid_bitfield,
	const GPUMemory<ObjectTransform>& world
) {
	auto res = m_render_buffer.out_resolution();
    uint32_t n_elements = product(res);
	if (m_reset_color_buffer) {
		init_rays_from_camera(
			sample_index,
			focal_length, 
			view.camera0, 
			screen_center,
			snap_to_pixel_centers
		);
		m_reset_color_buffer = false;
	}
	std::vector<LightProbeData> probe_data;
	for (auto& probe : light_probes) {
		probe_data.push_back(probe.data());
	}
	GPUMemory<LightProbeData> g_light_probes;
	g_light_probes.resize_and_copy_from_host(probe_data);

	linear_kernel(raytrace, 0, m_stream_ray, n_elements,
		m_rays[0].origin,
		m_rays[0].dir,
		view.camera0,
		lights.data(),
		lights.size(),
		materials.data(),
		materials.size(),
		world.data(),
		world.size(),
		view.cone_angle_constant,
		static_cast<size_t>(m_n_steps),
		m_view_nerf_shadow,
		view.render_aabb,
		view.render_aabb_to_local,
		focal_length,
		density_grid_bitfield,
		view.min_mip,
		view.max_mip,
		m_buffer_to_show,
		static_cast<size_t>(m_samples),
		static_cast<size_t>(m_ray_iters),
		static_cast<size_t>(m_shadow_iters),
		m_attenuation_coeff,
		depth_offset,
		m_rand_state,
		m_rays[0].rgba,
		m_rays[0].depth
	);
	sync();
}

void RayTracer::overlay(CudaRenderBufferView nerf_scene, size_t syn_px_scale, ngp::EColorSpace color_space, ngp::ETonemapCurve tonemap_curve, float exposure) {
	ivec2 res = resolution();
	const dim3 threads = { 16, 8, 1 };
	const dim3 blocks = { div_round_up((uint32_t)res.x, threads.x), div_round_up((uint32_t)res.y, threads.y), 1 };
	overlay_nerf<<<blocks, threads, 0, m_stream_ray>>>(
		res, 
		static_cast<int>(syn_px_scale),
		m_render_buffer.frame_buffer(),
		m_render_buffer.depth_buffer(),
		m_rays[0].rgba,
		m_rays[0].depth,
		nerf_scene.frame_buffer,
		nerf_scene.depth_buffer,
		color_space,
		tonemap_curve,
		exposure
	);
	sync();
}

void RayTracer::load(std::vector<vec4>& frame_cpu, std::vector<float>& depth_cpu) {
    m_rgba_texture->load_gpu(m_render_buffer.frame_buffer(), resolution(), frame_cpu);
    m_depth_texture->load_gpu(m_render_buffer.depth_buffer(), resolution(), 1, depth_cpu);
	sync();
}

void RayTracer::imgui() {
	constexpr int img_filter_type_count = sizeof(img_filter_type_names) / sizeof(img_filter_type_names[0]);
	constexpr int img_buffer_type_count = sizeof(img_buffer_type_names) / sizeof(img_buffer_type_names[0]);
	if (ImGui::CollapsingHeader("Raytracer", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::Checkbox("View NeRF shadows on Virtual Objects", &m_view_nerf_shadow);
		ImGui::InputInt("Number of shadow steps", &m_n_steps);
		ImGui::InputInt("Samples", &m_samples);
		ImGui::InputInt("Bounces", &m_ray_iters);
		ImGui::InputInt("Shadow Samples", &m_shadow_iters);
		ImGui::InputFloat("Attenuation", &m_attenuation_coeff);
		ImGui::SetNextItemOpen(true);
		if (ImGui::TreeNode("Display Filter")) {
			if (ImGui::Combo("Filter Type", (int*)&m_filter_to_use, img_filter_type_names, img_filter_type_count)) {
				tlog::success() << "Filter type id changed to: " << img_filter_type_names[(size_t) m_filter_to_use];
			}
			if (ImGui::Combo("Buffer Type", (int*)&m_buffer_to_show, img_buffer_type_names, img_buffer_type_count)) {
				tlog::success() << "buffer type id changed to: " << img_buffer_type_names[(size_t) m_buffer_to_show];
			}
			ImGui::TreePop();
		}
	}
}

}