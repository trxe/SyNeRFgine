#include <synerfgine/raytracer.cuh>
#include <neural-graphics-primitives/nerf_device.cuh>

namespace sng {
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
	vec3* __restrict__ normal,
	int32_t* __restrict__ mat_idx,
	float* __restrict__ t,
	bool* __restrict__ alive
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

	ray.d = normalize(ray.d);

	if (!ray.is_valid()) {
		origin[idx] = ray.o;
		alive[idx] = false;
		return;
	}

	origin[idx] = ray.o;
	dir[idx] = ray.d;
	normal[idx] = vec3(0.0);
	t[idx] = 0.0;
	mat_idx[idx] = -1;
	alive[idx] = true;
}

__global__ void transform_payload(
	uint32_t n_elements,
	const vec3* __restrict__ src_origin,
	const vec3* __restrict__ src_dir,
	const vec3* __restrict__ src_normal,
	vec3* __restrict__ dst_origin,
	vec3* __restrict__ dst_dir,
	vec3* __restrict__ dst_normal,
	mat3 rotation,
	vec3 translation,
	float scale_val,
	bool o2w
) {
	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= n_elements) return;
	if (!o2w) { // world space to object space
		mat3 scale = scale_val * mat3::identity();
		rotation = scale * rotation;
		dst_origin[idx] = rotation * (src_origin[idx] + translation);
		dst_dir[idx] = rotation * src_dir[idx];
		dst_normal[idx] = rotation * src_normal[idx];
	} else { // object space to world space
		dst_origin[idx] = rotation * (src_origin[idx] * scale_val) + translation;
		dst_dir[idx] = rotation * src_dir[idx] * scale_val;
		dst_normal[idx] = rotation * src_normal[idx] * scale_val;
	}
}

__global__ void raytrace(uint32_t n_elements, 
	const vec3* __restrict__ src_positions, 
	const vec3* __restrict__ src_directions, 
	// vec3* __restrict__ next_positions, 
	// vec3* __restrict__ next_directions, 
	// vec3* __restrict__ normals, 
	// float* __restrict__ t, 
	// int32_t* __restrict__ mat_idx,
	// bool* __restrict__ alive,
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
	curandState_t* __restrict__ rand_state,
	LightProbeData* __restrict__ light_probes,
	vec4* __restrict__ frame_buffer,
	float* __restrict__ depth_buffer
) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) return;

	auto src_pos = src_positions[i];
	auto src_dir = src_directions[i];
	float d = MAX_DEPTH();
	vec3 next_pos = src_pos;
	vec3 next_dir = src_dir;
	vec3 N = vec3(0.0);
	int32_t material = -1;
	bool is_alive = false;
	int32_t obj_id = -1;

	// tracing
	for (size_t w = 0; w < world_count; ++w) {
		const ObjectTransform& obj = world[w];
		auto p = ngp::ray_intersect_nodes(src_pos, src_dir, obj.scale, obj.pos, obj.rot, obj.g_node, obj.g_tris);
		if (p.first >= 0 && p.second < d) {
			d = p.second;
			next_pos = src_pos + p.second * src_dir;
			N = normalize(obj.scale * (obj.rot * (obj.g_tris[p.first].normal() + obj.pos)));
			is_alive = materials[obj.mat_id].scatter(next_pos, N, next_dir, rand_state[i]);
			material = obj.mat_id;
			obj_id = w;
		}
	}
	/*
	next_positions[i] = next_pos;
	next_directions[i] = next_dir;
	t[i] = d;
	normals[i] = N;
	mat_idx[i] = material;
	alive[i] = is_alive;

	frame_buffer[i] = vec4(0.0);
	*/
	depth_buffer[i] = d;

	// shading
	vec3 V = -normalize(src_dir);
	vec3 col = is_alive ? materials[material].ka : vec3(0.0);
	for (size_t l = 0; l < light_count && is_alive; ++l) {
		const Light& light = lights[l];
		vec3 lightpos = light.sample(rand_state[i]);
		const float full_d = length2(lightpos - next_pos);
		const vec3 L = normalize(lightpos - next_pos);
		const vec3 invL = vec3(1.0f) / L;
		const vec3 R = reflect(L, N);

		vec4 tmp_refl_col{0.0};
		float tmp_refl_depth{0.0};
		LightProbeData& probe_data = light_probes[obj_id];
		sample_probe(probe_data.position, probe_data.resolution, next_pos, probe_data.rgba, probe_data.depth, tmp_refl_col, tmp_refl_depth);
		const Material& this_mat = materials[material];

		vec3 tmp_col = max(0.0f, dot(L, N)) * this_mat.kd * light.intensity + pow(max(0.0f, dot(R, V)), this_mat.n) * this_mat.ks;
		tmp_col = tmp_col * (1.0f - this_mat.rg) + tmp_refl_col.rgb() * this_mat.rg;
		float nerf_shadow = show_nerf_shadow ? 0.0 : full_d;
		for (uint32_t j = 0; j < n_steps && show_nerf_shadow; ++j) {
			nerf_shadow = if_unoccupied_advance_to_next_occupied_voxel(nerf_shadow, cone_angle_constant, {next_pos, L}, invL, density_grid, min_mip, max_mip, render_aabb, render_aabb_to_local);
			if (nerf_shadow >= full_d) {
				nerf_shadow = full_d;
				break;
			}
			float dt = calc_dt(nerf_shadow, cone_angle_constant);
			nerf_shadow += dt;
		}
		float shadow_depth = full_d;

		for (size_t w = 0; w < world_count; ++w) {
			if (w == obj_id) continue;
			ObjectTransform obj = world[w];
			auto [hit, d] = ngp::ray_intersect_nodes(src_pos, L, obj.scale, obj.pos, obj.rot, obj.g_node, obj.g_tris);
			if (hit >= 0) shadow_depth = min(d, shadow_depth);
		}
		col += smoothstep(min(nerf_shadow, shadow_depth) / full_d) * tmp_col;
	}
	switch (buffer_type) {
	case ImgBufferType::Normal:
		frame_buffer[i].rgb() = vec3_to_col(N);
		break;
	case ImgBufferType::Direction:
		frame_buffer[i].rgb() = vec3_to_col(next_dir);
		break;
	case ImgBufferType::Origin:
		frame_buffer[i].rgb() = vec3_to_col(next_pos);
		break;
	default:
		frame_buffer[i].rgb() = col;
		break;
	}
}

__global__ void shade_color(
	uint32_t n_elements,
	uint32_t obj_id,
	// for Syn Obj
	const vec3* __restrict__ world_origin,
	const vec3* __restrict__ world_dir,
	const vec3* __restrict__ world_normal,
	const int32_t* __restrict__ mat_idxs,
	const float* __restrict__ depths,
	const bool* __restrict__ is_alive,
	Light* __restrict__ world_lights,
	uint32_t light_count,
	Material* __restrict__ materials,
	uint32_t material_count,
	ObjectTransform* __restrict__ obj_transforms,
	uint32_t object_count,
	// for NeRF shadows
	bool show_nerf_shadow,
	BoundingBox render_aabb,
	mat3 render_aabb_to_local,
	ivec2 focal_length,
	uint32_t n_steps,
	const uint8_t* __restrict__ density_grid,
	uint32_t min_mip,
	uint32_t max_mip,
	float cone_angle_constant,
	curandState_t* rand_state,
	vec3* __restrict__ out_color,
	float* __restrict__ out_depth
) {
	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= n_elements) return;
	int32_t mat_id = mat_idxs[idx];
	out_depth[idx] = depths[idx];

	if (mat_id < 0 || mat_id >= material_count || !is_alive[idx]) return;

	vec3 pos = world_origin[idx];
	vec3 N = normalize(world_normal[idx]);
	vec3 V = -normalize(world_dir[idx]);
	Material* mat_p = materials+mat_id;
	out_color[idx] = mat_p->ka;
	for (size_t i = 0; i < light_count; ++i) {
		const Light& light = world_lights[i];
		vec3 lightpos = light.sample(rand_state[idx]);
		// const float full_d = length2(light.pos - pos);
		// const vec3 L = normalize(light.pos - pos);
		const float full_d = length2(lightpos - pos);
		const vec3 L = normalize(lightpos - pos);
		const vec3 invL = vec3(1.0f) / L;
		const vec3 R = reflect(L, N);
		vec3 tmp_col = max(0.0f, dot(L, N)) * mat_p->kd * light.intensity + pow(max(0.0f, dot(R, V)), mat_p->n) * mat_p->ks;
		float nerf_shadow = show_nerf_shadow ? 0.0 : full_d;
		for (uint32_t j = 0; j < n_steps && show_nerf_shadow; ++j) {
			nerf_shadow = if_unoccupied_advance_to_next_occupied_voxel(nerf_shadow, cone_angle_constant, {pos, L}, invL, density_grid, min_mip, max_mip, render_aabb, render_aabb_to_local);
			if (nerf_shadow >= full_d) {
				nerf_shadow = full_d;
				break;
			}
			float dt = calc_dt(nerf_shadow, cone_angle_constant);
			nerf_shadow += dt;
		}
		float shadow_depth = full_d;

		for (size_t t = 0; t < object_count; ++t) {
			if (t == obj_id) continue;
			ObjectTransform obj = obj_transforms[t];
			auto [hit, d] = ngp::ray_intersect_nodes(pos, L, obj.scale, obj.pos, obj.rot, obj.g_node, obj.g_tris);
			if (hit >= 0) shadow_depth = min(d, shadow_depth);
		}
		out_color[idx] += smoothstep(min(nerf_shadow, shadow_depth) / full_d) * tmp_col;
	}
}

__global__ void transfer_color(
	uint32_t n_elements,
	const vec3* __restrict__ src_color,
	const float* __restrict__ src_depth,
	vec4* __restrict__ frame_buffer,
	float* __restrict__ depth_buffer
) {
	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= n_elements || src_color == nullptr || src_depth == nullptr) return;
	vec3 scol = src_color[idx];
	float sdep = src_depth[idx];
	float currdep = depth_buffer[idx];
	if (sdep < currdep) {
		frame_buffer[idx] = vec4(scol, 1.0);
		depth_buffer[idx] = sdep;
	}
}

void RayTracer::enlarge(const ivec2& res) {
	if (m_render_buffer.out_resolution() == res) return;
    m_render_buffer.resize(res);
    size_t n_elements = product(res);
	n_elements = next_multiple(n_elements, size_t(BATCH_SIZE_GRANULARITY));
	auto scratch = allocate_workspace_and_distribute<
		vec3, float, vec3, vec3, vec3, int32_t, float, bool, // m_rays[0]
		vec3, float, vec3, vec3, vec3, int32_t, float, bool, // m_rays[1]
		// vec3, float, vec3, vec3, vec3, int32_t, float, bool, // m_rays_hit

		curandState_t,
		uint32_t,
		uint32_t
	>(
		m_stream_ray, &m_scratch_alloc,
		n_elements, n_elements, n_elements, n_elements, n_elements, n_elements, n_elements, n_elements,
		n_elements, n_elements, n_elements, n_elements, n_elements, n_elements, n_elements, n_elements,
		// n_elements, n_elements, n_elements, n_elements, n_elements, n_elements, n_elements, n_elements,
		n_elements,
		32, // 2 full cache lines to ensure no overlap
		32  // 2 full cache lines to ensure no overlap
	);

	m_rays[0].set(std::get<0>(scratch), std::get<1>(scratch), std::get<2>(scratch), std::get<3>(scratch), std::get<4>(scratch), std::get<5>(scratch), std::get<6>(scratch), std::get<7>(scratch), n_elements);
	m_rays[1].set(std::get<8>(scratch), std::get<9>(scratch), std::get<10>(scratch), std::get<11>(scratch), std::get<12>(scratch), std::get<13>(scratch), std::get<14>(scratch), std::get<15>(scratch), n_elements);
	// m_rays_hit.set(std::get<16>(scratch), std::get<17>(scratch), std::get<18>(scratch), std::get<19>(scratch), std::get<20>(scratch), std::get<21>(scratch), std::get<22>(scratch), std::get<23>(scratch), n_elements);

	m_rand_state = std::get<16>(scratch);
	m_hit_counter = std::get<17>(scratch);
	m_alive_counter = std::get<18>(scratch);
	// m_rand_state = std::get<24>(scratch);
	// m_hit_counter = std::get<25>(scratch);
	// m_alive_counter = std::get<26>(scratch);
	sync();
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
		m_rays[0].normal,
		m_rays[0].mat_idx,
		m_rays[0].t,
		m_rays[0].alive
	);
	sync();

	// m_n_rays_initialized = res.x * res.y;

	CUDA_CHECK_THROW(cudaMemsetAsync(m_rays[1].rgb, 0, m_n_rays_initialized * sizeof(vec3), m_stream_ray));
	CUDA_CHECK_THROW(cudaMemsetAsync(m_rays[1].depth, 0, m_n_rays_initialized * sizeof(float), m_stream_ray));
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
	bool snap_to_pixel_centers,
	const uint8_t* density_grid_bitfield,
	const GPUMemory<ObjectTransform>& world
) {
	auto res = m_render_buffer.out_resolution();
    uint32_t n_elements = product(res);
	init_rays_from_camera(
		sample_index,
		focal_length, 
		view.camera0, 
		screen_center,
		snap_to_pixel_centers
	);
	std::vector<LightProbeData> probe_data;
	for (auto& probe : light_probes) {
		probe_data.push_back(probe.data());
	}
	GPUMemory<LightProbeData> g_light_probes;
	g_light_probes.resize_and_copy_from_host(probe_data);

	linear_kernel(raytrace, 0, m_stream_ray, n_elements,
		m_rays[0].origin,
		m_rays[0].dir,
		// m_rays[1].origin,
		// m_rays[1].dir,
		// m_rays[1].normal,
		// m_rays[1].t,
		// m_rays[1].mat_idx,
		// m_rays[1].alive,
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
		m_rand_state,
		g_light_probes.data(),
		m_render_buffer.frame_buffer(),
		m_render_buffer.depth_buffer()
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
	if (ImGui::CollapsingHeader("Raytracer", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::Checkbox("View NeRF shadows on Virtual Objects", &m_view_nerf_shadow);
		ImGui::InputInt("Number of shadow steps", &m_n_steps);
		ImGui::SetNextItemOpen(true);
		if (ImGui::TreeNode("Display Filter")) {
			if (ImGui::Combo("Filter Type", (int*)&m_filter_to_use, img_filter_type_names, img_filter_type_count)) {
				tlog::success() << "Filter type id changed to: " << (int) m_filter_to_use;
			}
			ImGui::TreePop();
		}
	}
}

}