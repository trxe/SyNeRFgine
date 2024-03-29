#include <synerfgine/raytracer.cuh>
#include <neural-graphics-primitives/nerf_device.cuh>

namespace sng {
__device__ vec4 shade_object(const vec3& wi, SampledRay& ray, HitRecord& hit_info,
	Light* __restrict__ lights, const size_t& light_count, 
	ObjectTransform* __restrict__ objects, const size_t& object_count, 
	Material* __restrict__ materials, const size_t& material_count,
	curandState_t& rand_state
) {
	if (hit_info.material_idx < 0) return vec4(0.0);
	Material& material = materials[hit_info.material_idx];
	vec4 color{material.ka, 1.0};
	for (size_t l = 0; l < light_count; ++l) {
		Light& light = lights[l];
		vec3 lpos = light.sample(rand_state);
		vec3 L = normalize(lpos = hit_info.pos);
		int32_t obj_hit = -1; 
		depth_test_world(hit_info.pos, L, objects, object_count, obj_hit);
		if (obj_hit < 0 || obj_hit == hit_info.object_idx) {
			continue;
		}
		vec3 R = reflect(L, hit_info.normal);
		vec3 V = normalize(-wi);
		color.rgb() += material.local_color(L, hit_info.normal, R, wi, light);
	}
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
	size_t ray_iter_count,
	size_t shadow_count,
	float attenuation_coeff,
	curandState_t* __restrict__ rand_state,
	LightProbeData* __restrict__ light_probes,
	vec4* __restrict__ frame_buffer,
	float* __restrict__ depth_buffer
) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) return;
	constexpr const float MIN_DIST_T = 0.0001f;

	auto src_pos = vec3(0.0);
	auto src_dir = vec3(0.0);
	vec3 next_pos = src_positions[i];
	vec3 next_dir = src_directions[i];
	vec3 normal{0.0};
	bool is_alive = false;
	vec3 shade{0.0};
	float attenuation = attenuation_coeff;

	for (size_t iter = 0; iter < ray_iter_count; ++iter) {
		src_pos = next_pos;
		src_dir = next_dir;
		next_dir = next_pos = vec3(0.0);
		int32_t obj_id = -1;
		int32_t material = -1;
		float d = MAX_DEPTH();
		vec3 N = vec3(0.0);
		is_alive = false;
		// tracing
		for (size_t w = 0; w < world_count; ++w) {
			const ObjectTransform& obj = world[w];
			auto p = ngp::ray_intersect_nodes(src_pos, src_dir, obj.scale, obj.pos, obj.rot, obj.g_node, obj.g_tris);
			if (p.first >= 0 && p.second < d && p.second > MIN_DIST_T) {
				d = p.second;
				next_pos = src_pos + p.second * src_dir;
				N = normalize(obj.rot * obj.g_tris[p.first].normal());
				const Material& mat = materials[obj.mat_id];
				is_alive = mat.scatter(N, src_dir, next_dir, rand_state[i]);
				vec3 r = next_dir;
				next_dir = obj.g_tris[p.first].scatter(next_dir, (1.0 - mat.rg) * ngp::PI() * 2.0f, rand_state[i]);
				material = obj.mat_id;
				obj_id = w;
			}
		}

		if (!is_alive) break;
		if (!iter) depth_buffer[i] = d;

		// shading
		vec3 V = -normalize(src_dir);
		const Material& this_mat = materials[material];
		vec3 col = this_mat.ka;
		for (size_t l = 0; l < light_count && is_alive; ++l) {
			const Light& light = lights[l];
			float shadow_amount = 0.0;
			for (size_t spp_shadow = 0; spp_shadow < shadow_count; ++spp_shadow) {
				vec3 lightpos = light.sample(rand_state[i]);
				const float full_d = length2(lightpos - next_pos);
				const vec3 L = normalize(lightpos - next_pos);
				const vec3 invL = vec3(1.0f) / L;
				const vec3 R = reflect(L, N);

				vec4 tmp_refl_col{0.0};
				float tmp_refl_depth{0.0};
				vec3 diffuse = this_mat.kd;
				if (light_probes) {
					LightProbeData& probe_data = light_probes[obj_id];
					// sample_probe(probe_data.position, probe_data.resolution, next_pos, probe_data.rgba, probe_data.depth, tmp_refl_col, tmp_refl_depth);
					sample_probe_dir(next_dir, probe_data.resolution, probe_data.rgba, probe_data.depth, tmp_refl_col, tmp_refl_depth);
					tmp_refl_col.rgb() = tmp_refl_depth < MAX_DEPTH() ? tmp_refl_col.rgb() : vec3(0.0);

					diffuse = tmp_refl_col.rgb() * this_mat.rg + this_mat.kd * (1.0f - this_mat.rg);
				}
				vec3 tmp_col = max(0.0f, dot(L, N)) * diffuse * light.intensity + pow(max(0.0f, dot(R, V)), this_mat.n) * this_mat.ks;

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
					auto [hit, d] = ngp::ray_intersect_nodes(next_pos, L, obj.scale, obj.pos, obj.rot, obj.g_node, obj.g_tris);
					if (hit >= 0 && d >= MIN_DIST_T) shadow_depth = min(d, shadow_depth);
				}
				col += smoothstep(min(nerf_shadow, shadow_depth) / (full_d * (float)shadow_count)) * tmp_col;
			}
		}
		shade += col * attenuation;
		attenuation *= attenuation_coeff * this_mat.rg;
		normal = N;
	}
	switch (buffer_type) {
	case ImgBufferType::Normal:
		// frame_buffer[i].rgb() = vec3_to_col(normal);
		frame_buffer[i].rgb() = normal;
		break;
	case ImgBufferType::NextDirection:
		// frame_buffer[i].rgb() = vec3_to_col(next_dir);
		frame_buffer[i].rgb() = next_dir;
		break;
	case ImgBufferType::SrcDirection:
		// frame_buffer[i].rgb() = vec3_to_col(src_dir);
		frame_buffer[i].rgb() = src_dir;
		break;
	case ImgBufferType::NextOrigin:
		frame_buffer[i].rgb() = vec3_to_col(next_pos);
		break;
	case ImgBufferType::SrcOrigin:
		frame_buffer[i].rgb() = vec3_to_col(src_pos);
		break;
	default:
		frame_buffer[i].rgb() = shade;
		break;
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
	const GPUMemory<ObjectTransform>& world,
	bool enable_reflection
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
		static_cast<size_t>(m_ray_iters),
		static_cast<size_t>(m_shadow_iters),
		m_attenuation_coeff,
		m_rand_state,
		enable_reflection ? g_light_probes.data() : nullptr,
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
	constexpr int img_buffer_type_count = sizeof(img_buffer_type_names) / sizeof(img_buffer_type_names[0]);
	if (ImGui::CollapsingHeader("Raytracer", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::Checkbox("View NeRF shadows on Virtual Objects", &m_view_nerf_shadow);
		ImGui::InputInt("Number of shadow steps", &m_n_steps);
		ImGui::InputInt("Ray iters", &m_ray_iters);
		ImGui::InputInt("Shadow iters", &m_shadow_iters);
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