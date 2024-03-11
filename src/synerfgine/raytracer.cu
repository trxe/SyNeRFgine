#include <synerfgine/raytracer.cuh>

namespace sng {

void RayTracer::enlarge(const ivec2& res) {
    m_render_buffer.resize(res);
    size_t n_elements = product(res);
	n_elements = next_multiple(n_elements, size_t(BATCH_SIZE_GRANULARITY));
	auto scratch = allocate_workspace_and_distribute<
		vec4, float, RayPayload, // m_rays[0]
		vec4, float, RayPayload, // m_rays[1]
		vec4, float, RayPayload, // m_rays_hit

		uint32_t,
		uint32_t
	>(
		m_stream_ray, &m_scratch_alloc,
		n_elements, n_elements, n_elements,
		n_elements, n_elements, n_elements,
		n_elements, n_elements, n_elements,
		32, // 2 full cache lines to ensure no overlap
		32  // 2 full cache lines to ensure no overlap
	);

	m_rays[0].set(std::get<0>(scratch), std::get<1>(scratch), std::get<2>(scratch), n_elements);
	m_rays[1].set(std::get<3>(scratch), std::get<4>(scratch), std::get<5>(scratch), n_elements);
	m_rays_hit.set(std::get<6>(scratch), std::get<7>(scratch), std::get<8>(scratch), n_elements);

	m_hit_counter = std::get<9>(scratch);
	m_alive_counter = std::get<10>(scratch);
	sync();
}

void RayTracer::test() {
	auto res = m_render_buffer.out_resolution();
    uint32_t n_elements = product(res);
    linear_kernel(debug_uv_shade, 0, m_stream_ray, n_elements, m_render_buffer.frame_buffer(), m_render_buffer.depth_buffer(), res);
	sync();
}

void RayTracer::load(std::vector<vec4>& frame_cpu, std::vector<float>& depth_cpu) {
    m_rgba_texture->load_gpu(m_render_buffer.frame_buffer(), resolution(), frame_cpu);
    m_depth_texture->load_gpu(m_render_buffer.depth_buffer(), resolution(), 1, depth_cpu);
}

}