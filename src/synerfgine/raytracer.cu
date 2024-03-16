#include <synerfgine/raytracer.cuh>
#include <neural-graphics-primitives/nerf_device.cuh>

namespace sng {
__global__ void test_intersect_aabb(
	uint32_t n_elements,
	BoundingBox aabb,
	vec3* __restrict__ origin,
	vec3* __restrict__ dir,
	vec3* __restrict__ normal,
	int32_t* __restrict__ mat_idx,
	float* __restrict__ t,
	bool* __restrict__ alive
) {
	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= n_elements) return;
	vec2 ts = aabb.ray_intersect(origin[idx], dir[idx]);
	t[idx] = ts.x;
	if (ts.x < 10000.0) {
		normal[idx] = normalize(vec3(ts.x, ts.y, 0.0));
	} else {
		normal[idx] = vec3(0.0, 0.0, 0.0);
	}
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

	// DEBUGGER
	// frame_buffer[idx].rgb() = dir[idx];
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
	float scale,
	bool o2w
) {
	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= n_elements) return;
	if (!o2w) { // world space to object space
		rotation = scale * rotation;
		dst_origin[idx] = rotation * (src_origin[idx] + translation);
		dst_dir[idx] = rotation * src_dir[idx];
		dst_normal[idx] = rotation * src_normal[idx];
	} else { // object space to world space
		dst_origin[idx] = rotation * (src_origin[idx] * scale) + translation;
		dst_dir[idx] = rotation * src_dir[idx] * scale;
		dst_normal[idx] = rotation * src_normal[idx] * scale;
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
		Light* light_p = world_lights+i;
		float full_d = length2(light_p->pos - pos);
		vec3 L = normalize(light_p->pos - pos);
		vec3 invL = vec3(1.0f) / L;
		float cone_angle = calc_cone_angle(dot(L, L), focal_length, cone_angle_constant);
		vec3 R = reflect(L, N);
		vec3 tmp_col = max(0.0f, dot(L, N)) * mat_p->kd + pow(max(0.0f, dot(R, V)), mat_p->n) * mat_p->ks;
		float nerf_shadow = 0.0;
		for (uint32_t j = 0; j < n_steps && show_nerf_shadow; ++j) {
			nerf_shadow = if_unoccupied_advance_to_next_occupied_voxel(nerf_shadow, cone_angle, {pos, L}, invL, density_grid, min_mip, max_mip, render_aabb, render_aabb_to_local);
			if (nerf_shadow >= full_d) {
				nerf_shadow = full_d;
				break;
			}
			float dt = calc_dt(nerf_shadow, cone_angle);
			nerf_shadow += dt;
		}
		float shadow_depth = full_d;

		for (size_t t = 0; t < object_count; ++t) {
			if (t == obj_id) continue;
			ObjectTransform obj = obj_transforms[t];
			pos = inverse(obj.rot) / obj.scale * (pos - obj.pos);
			L = inverse(obj.rot) / obj.scale * L;
			auto [hit, d] = ngp::ray_intersect_nodes(pos, L, obj.g_node, obj.g_tris);
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
    m_render_buffer.resize(res);
    size_t n_elements = product(res);
	n_elements = next_multiple(n_elements, size_t(BATCH_SIZE_GRANULARITY));
	auto scratch = allocate_workspace_and_distribute<
		vec3, float, vec3, vec3, vec3, int32_t, float, bool, // m_rays[0]
		vec3, float, vec3, vec3, vec3, int32_t, float, bool, // m_rays[1]
		vec3, float, vec3, vec3, vec3, int32_t, float, bool, // m_rays_hit

		uint32_t,
		uint32_t
	>(
		m_stream_ray, &m_scratch_alloc,
		n_elements, n_elements, n_elements, n_elements, n_elements, n_elements, n_elements, n_elements,
		n_elements, n_elements, n_elements, n_elements, n_elements, n_elements, n_elements, n_elements,
		n_elements, n_elements, n_elements, n_elements, n_elements, n_elements, n_elements, n_elements,
		32, // 2 full cache lines to ensure no overlap
		32  // 2 full cache lines to ensure no overlap
	);

	m_rays[0].set(std::get<0>(scratch), std::get<1>(scratch), std::get<2>(scratch), std::get<3>(scratch), std::get<4>(scratch), std::get<5>(scratch), std::get<6>(scratch), std::get<7>(scratch), n_elements);
	m_rays[1].set(std::get<8>(scratch), std::get<9>(scratch), std::get<10>(scratch), std::get<11>(scratch), std::get<12>(scratch), std::get<13>(scratch), std::get<14>(scratch), std::get<15>(scratch), n_elements);
	m_rays_hit.set(std::get<16>(scratch), std::get<17>(scratch), std::get<18>(scratch), std::get<19>(scratch), std::get<20>(scratch), std::get<21>(scratch), std::get<22>(scratch), std::get<23>(scratch), n_elements);

	m_hit_counter = std::get<24>(scratch);
	m_alive_counter = std::get<25>(scratch);
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
	std::vector<Material>& h_materials, 
	std::vector<VirtualObject>& h_vo, 
	std::vector<Light>& h_light, 
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
	set_gpu_materials_light(h_materials, h_light);
    // linear_kernel(debug_uv_shade, 0, m_stream_ray, n_elements, m_render_buffer.frame_buffer(), m_render_buffer.depth_buffer(), res);
	init_rays_from_camera(
		sample_index,
		focal_length, 
		view.camera0, 
		screen_center,
		snap_to_pixel_centers
	);
	for (auto& obj : h_vo) {
		std::shared_ptr<TriangleBvh> bvh = obj.bvh();
		auto& o2w_rot = obj.get_rotate();
		auto& o2w_trans = obj.get_translate();
		float o2w_scale = obj.get_scale();
		linear_kernel(transform_payload, 0, m_stream_ray, n_elements,
			m_rays[0].origin,
			m_rays[0].dir,
			m_rays[0].normal,
			m_rays[1].origin,
			m_rays[1].dir,
			m_rays[1].normal,
			inverse(o2w_rot),
			-o2w_trans,
			1.0/o2w_scale,
			false
		);
		sync();
		bvh->ray_trace_gpu(
			n_elements,
			m_rays[1].origin,
			m_rays[1].dir,
			m_rays[1].normal,
			m_rays[1].t,
			m_rays[1].mat_idx,
			m_rays[1].alive,
			obj.get_mat_idx(),
			obj.gpu_triangles(),
			m_stream_ray
		);
		sync();
		linear_kernel(transform_payload, 0, m_stream_ray, n_elements,
			m_rays[1].origin,
			m_rays[1].dir,
			m_rays[1].normal,
			m_rays[1].origin,
			m_rays[1].dir,
			m_rays[1].normal,
			o2w_rot,
			o2w_trans,
			o2w_scale,
			true
		);
		sync();
		linear_kernel(shade_color, 0, m_stream_ray, n_elements,
			obj.get_id(),
			m_rays[1].origin,
			m_rays[1].dir,
			m_rays[1].normal,
			m_rays[1].mat_idx,
			m_rays[1].t,
			m_rays[1].alive,
			d_lights.data(),
			d_lights.size(),
			d_materials.data(),
			d_materials.size(),
			world.data(),
			world.size(),
			m_view_nerf_shadow,
			view.render_aabb,
			view.render_aabb_to_local,
			focal_length,
			static_cast<size_t>(m_n_steps),
			density_grid_bitfield,
			view.min_mip,
			view.max_mip,
			view.cone_angle_constant,
			m_rays[1].rgb,
			m_rays[1].depth
		);
		sync();
		linear_kernel(transfer_color, 0, m_stream_ray, n_elements,
			buffer_selector(m_rays[1], m_buffer_to_show),
			m_rays[1].depth,
			m_render_buffer.frame_buffer(),
			m_render_buffer.depth_buffer()
		);
		sync();
	}
	sync();
}

void RayTracer::load(std::vector<vec4>& frame_cpu, std::vector<float>& depth_cpu) {
    m_rgba_texture->load_gpu(m_render_buffer.frame_buffer(), resolution(), frame_cpu);
    m_depth_texture->load_gpu(m_render_buffer.depth_buffer(), resolution(), 1, depth_cpu);
	sync();
}

void RayTracer::imgui() {
	// constexpr int img_buffer_type_count = sizeof(img_buffer_type_names) / sizeof(img_buffer_type_names[0]);
	if (ImGui::CollapsingHeader("Raytracer", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::Checkbox("View NeRF shadows on Virtual Objects", &m_view_nerf_shadow);
		ImGui::InputInt("Number of shadow steps", &m_n_steps);
		// ImGui::SetNextItemOpen(true);
		// if (ImGui::TreeNode("Display Buffer")) {
		// 	if (ImGui::Combo("Buffer Type", (int*)&m_buffer_to_show, img_buffer_type_names, img_buffer_type_count)) {
		// 	}
		// 	ImGui::TreePop();
		// }
	}
}

}