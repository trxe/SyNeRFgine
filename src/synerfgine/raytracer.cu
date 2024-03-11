#include <synerfgine/raytracer.cuh>
#include <iostream>

namespace sng {
__global__ void init_rays_with_payload_kernel_nerf(
	uint32_t sample_index,
	ivec2 resolution,
	vec2 focal_length,
	mat4x3 camera_matrix0,
	mat4x3 camera_matrix1,
	vec4 rolling_shutter,
	vec2 screen_center,
	bool snap_to_pixel_centers,
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

	vec2 pixel_offset = ld_random_pixel_offset(snap_to_pixel_centers ? 0 : sample_index);
	vec2 uv = vec2{(float)x + pixel_offset.x, (float)y + pixel_offset.y} / vec2(resolution);
	mat4x3 camera = get_xform_given_rolling_shutter({camera_matrix0, camera_matrix1}, rolling_shutter, uv, ld_random_val(sample_index, idx * 72239731));

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
	vec3 translation
) {
	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= n_elements) return;
	dst_origin[idx] = rotation * src_origin[idx] + translation;
	dst_dir[idx] = rotation * src_dir[idx];
	dst_normal[idx] = rotation * src_normal[idx];
}

__global__ void transfer_color(
	uint32_t n_elements,
	const vec3* __restrict__ src,
	vec4* __restrict__ frame_buffer
) {
	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= n_elements) return;
	frame_buffer[idx].rgb() = normalize(src[idx]) * 0.5f + 0.5f;
}

void RayTracer::enlarge(const ivec2& res) {
    m_render_buffer.resize(res);
    size_t n_elements = product(res);
	n_elements = next_multiple(n_elements, size_t(BATCH_SIZE_GRANULARITY));
	auto scratch = allocate_workspace_and_distribute<
		vec4, float, vec3, vec3, vec3, int32_t, float, bool, // m_rays[0]
		vec4, float, vec3, vec3, vec3, int32_t, float, bool, // m_rays[1]
		vec4, float, vec3, vec3, vec3, int32_t, float, bool, // m_rays_hit

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
	const mat4x3& camera_matrix0,
	const mat4x3& camera_matrix1,
	const vec4& rolling_shutter,
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
		camera_matrix0,
		camera_matrix1,
		rolling_shutter,
		screen_center,
		snap_to_pixel_centers,
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

	// CUDA_CHECK_THROW(cudaMemsetAsync(m_rays[0].rgba, 0, m_n_rays_initialized * sizeof(vec4), m_stream_ray));
	// CUDA_CHECK_THROW(cudaMemsetAsync(m_rays[0].depth, 0, m_n_rays_initialized * sizeof(float), m_stream_ray));
}

void RayTracer::render(
	const std::vector<Material>& h_materials, 
	std::vector<VirtualObject>& h_vo, 
	const Testbed::View& view, 
	const vec2& screen_center,
	uint32_t sample_index,
	const vec2& focal_length,
	bool snap_to_pixel_centers
) {
	auto res = m_render_buffer.out_resolution();
    uint32_t n_elements = product(res);
    // linear_kernel(debug_uv_shade, 0, m_stream_ray, n_elements, m_render_buffer.frame_buffer(), m_render_buffer.depth_buffer(), res);
	init_rays_from_camera(
		sample_index,
		focal_length, 
		view.camera0, 
		view.camera1,
		view.rolling_shutter,
		screen_center,
		snap_to_pixel_centers
	);
	for (auto& obj : h_vo) {
		std::shared_ptr<TriangleBvh> bvh = obj.bvh();
		auto& o2w_rot = obj.get_rotate();
		auto& o2w_trans = obj.get_translate();
		std::cout << 
			o2w_trans.x << " " <<
			o2w_trans.y << " " <<
			o2w_trans.z << std::endl;
		linear_kernel(transform_payload, 0, m_stream_ray, n_elements,
			m_rays[0].origin,
			m_rays[0].dir,
			m_rays[0].normal,
			m_rays[1].origin,
			m_rays[1].dir,
			m_rays[1].normal,
			inverse(o2w_rot),
			-o2w_trans
		);
		sync();
		bvh->ray_trace_gpu(
			n_elements,
			m_rays[1].origin,
			m_rays[1].dir,
			obj.gpu_triangles(),
			m_stream_ray
		);
		sync();
		linear_kernel(transfer_color, 0, m_stream_ray, n_elements,
			m_rays[1].dir,
			m_render_buffer.frame_buffer()
		);
		sync();
	}
	sync();
}

void RayTracer::load(std::vector<vec4>& frame_cpu, std::vector<float>& depth_cpu) {
    m_rgba_texture->load_gpu(m_render_buffer.frame_buffer(), resolution(), frame_cpu);
    m_depth_texture->load_gpu(m_render_buffer.depth_buffer(), resolution(), 1, depth_cpu);
}

}