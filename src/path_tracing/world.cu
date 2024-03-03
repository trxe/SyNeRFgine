#include <neural-graphics-primitives/path-tracing/world.cuh>

namespace ngp {
namespace pt {

using ngp::Ray;

__global__ void pt_debug_world_triangles(uint32_t n_elements, Tri* triangles) {
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_elements) return;
    Tri& triangle = triangles[idx];
    printf("triangle %d: %d [%f, %f, %f], [%f, %f, %f], [%f, %f, %f]\n", 
        idx, triangle.material_idx,
        triangle.point[0].x, triangle.point[0].y, triangle.point[0].z, 
        triangle.point[1].x, triangle.point[1].y, triangle.point[1].z, 
        triangle.point[2].x, triangle.point[2].y, triangle.point[2].z);
}

__global__ void pt_debug_mat(Material* d_mat) {
  d_mat->test();
}

__global__ void pt_reset_buffers(uint32_t n_elements, 
    vec3* __restrict__ beta,
    vec3* __restrict__ attenuation,
    vec3* __restrict__ aggregation,
    vec3* __restrict__ pos,
    float* __restrict__ shadow,
    Ray* __restrict__ ray,
    Ray* __restrict__ prev_ray,
    bool* __restrict__ end,
    bool* __restrict__ prev_end,
	HitRecord* __restrict__ hit_record
) {
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_elements) return;
	beta[idx] = vec3(0.0f);
	attenuation[idx] = vec3(0.0f);
	aggregation[idx] = vec3(0.0f);
	pos[idx] = vec3(0.0f);
	shadow[idx] = 0.0f;
	ray[idx] = Ray();
	prev_ray[idx] = Ray();
	end[idx] = false;
	prev_end[idx] = false;
	hit_record[idx] = HitRecord{};
}

__global__ void pt_debug_fb(uint32_t n_elements, ivec2 resolution, vec4* __restrict__ frame_buffer) {
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_elements) return;
    float x = (float)(idx % resolution.x) / (float) resolution.x;
    float y = (float)(idx / resolution.x) / (float) resolution.y;
    frame_buffer[idx] = vec4{x, y, 0.0f, 1.0f};
}

__global__ void pt_trace_once(uint32_t n_elements, 
	ivec2 resolution, 
	vec4* __restrict__ frame_buffer,
	uint32_t material_count,
	Material** __restrict__ w_material_gpu,
	uint32_t hittable_count,
	Hittable** __restrict__ w_hittable_gpu,
	Ray* __restrict__ rays,
	HitRecord* __restrict__ hit_records
) {
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_elements) return;
	vec3 rgb{};
	Ray& ray = rays[idx];
	HitRecord& rec = hit_records[idx];
	// for (size_t t = 0; t < hittable_count; ++t) {
	// 	Hittable* hittable = w_hittable_gpu[t];
	// 	if (!hittable) {
	// 		return;
	// 	}
	// 	vec3 c  = hittable->center();
	// 	rgb = c * 0.5f + vec3(0.5f);
	// 	// if (hittable->hit(ray, PT_EPSILON, PT_MAX_FLOAT, rec)) {
	// 	// 	rgb = normalize(rec.pos);
	// 	// 	rgb = rgb * 0.5f + vec3(0.5f);
	// 	// }
	// }
	for (size_t t = 0; t < material_count; ++t) {
		Material* mat = w_material_gpu[t];
		if (idx % 100 == 0) mat->test();
	}
    frame_buffer[idx] = vec4{rgb, 1.0f};
}

__global__ void init_rays_world_kernel_nerf(
	uint32_t sample_index,
	ivec2 resolution,
	vec2 focal_length,
	mat4x3 camera_matrix0,
	mat4x3 camera_matrix1,
	vec4 rolling_shutter,
	vec2 screen_center,
	vec3 parallax_shift,
    ngp::BoundingBox render_aabb,
    mat3 render_aabb_to_local,
	bool snap_to_pixel_centers,
	float near_distance,
	float plane_z,
	float aperture_size,
	Foveation foveation,
	Lens lens,
	Buffer2DView<const uint8_t> hidden_area_mask,
	Buffer2DView<const vec2> distortion,
    Ray* __restrict__ rays,
    bool* __restrict__ ends
) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= resolution.x || y >= resolution.y) {
		return;
	}

	uint32_t idx = x + resolution.x * y;

	if (plane_z < 0) {
		aperture_size = 0.0;
	}

	vec2 pixel_offset = ld_random_pixel_offset(snap_to_pixel_centers ? 0 : sample_index);
	vec2 uv = vec2{(float)x + pixel_offset.x, (float)y + pixel_offset.y} / vec2(resolution);
	mat4x3 camera = get_xform_given_rolling_shutter({camera_matrix0, camera_matrix1}, rolling_shutter, uv, ld_random_val(sample_index, idx * 72239731));

	Ray ray = uv_to_ray(
		sample_index,
		uv,
		resolution,
		focal_length,
		camera,
		screen_center,
		parallax_shift,
		near_distance,
		plane_z,
		aperture_size,
		foveation,
		hidden_area_mask,
		lens,
		distortion
	);

	float t = fmaxf(render_aabb.ray_intersect(render_aabb_to_local * ray.o, render_aabb_to_local * ray.d).x, 0.0f) + 1e-6f;
    ends[idx] = !ray.is_valid() || !render_aabb.contains(render_aabb_to_local * ray(t));
	if (!ends[idx]) rays[idx] = ray;
}

void World::render(
    cudaStream_t stream,
    const CudaRenderBufferView& render_buffer,
    const vec2& focal_length,
    const mat4x3& camera_matrix0,
    const mat4x3& camera_matrix1,
    const vec4& rolling_shutter,
    const vec2& screen_center,
    const Foveation& foveation,
    int visualized_dimension
) {
    const auto& resolution = render_buffer.resolution;
    auto n_elements = resolution.x * resolution.y;
    // linear_kernel(pt_debug_fb, 0, stream, n_elements, resolution, render_buffer.frame_buffer);
	linear_kernel(pt_trace_once, 0, stream, n_elements, 
		resolution, 
		render_buffer.frame_buffer,
		w_materials.size(),
		w_material_gpu,
		w_hittables.size(),
		w_hittable_gpu,
		px_ray,
		px_hit_record);
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
}

void World::init_rays(
    cudaStream_t stream,
    const uint32_t& sample_index,
    const ivec2& resolution,
    const vec2& focal_length,
    const mat4x3& camera_matrix0,
    const mat4x3& camera_matrix1,
    const vec4& rolling_shutter,
    const vec2& screen_center,
    const vec3& parallax_shift,
	const ngp::BoundingBox& render_aabb,
	const mat3& render_aabb_to_local,
    const bool& snap_to_pixel_centers,
    const float& near_distance,
    const float& plane_z,
    const float& aperture_size,
    const Foveation& foveation,
    const Lens& lens,
    Buffer2DView<const uint8_t> hidden_area_mask,
    Buffer2DView<const vec2> distortion
) {
	const dim3 threads = { 16, 8, 1 };
	const dim3 blocks = { div_round_up((uint32_t)resolution.x, threads.x), div_round_up((uint32_t)resolution.y, threads.y), 1 };
	linear_kernel(pt_reset_buffers, 0, stream, resolution.x * resolution.y, 
		px_beta,
		px_attenuation,
		px_aggregation,
		px_pos,
		px_shadow,
		px_ray,
		px_prev_ray,
		px_end,
		px_prev_end,
		px_hit_record
	);
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

    init_rays_world_kernel_nerf<<<blocks, threads, 0, stream>>> (
		sample_index,
		resolution,
		focal_length,
		camera_matrix0,
		camera_matrix1,
		rolling_shutter,
		screen_center,
		parallax_shift,
		render_aabb,
		render_aabb_to_local,
		snap_to_pixel_centers,
		near_distance,
		plane_z,
		aperture_size,
		foveation,
		lens,
		hidden_area_mask,
		distortion,
        px_ray,
        px_end);
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
}

}
}