#include <synerfgine/probe.cuh>

namespace sng {

#define M_PI 3.1419267f

__device__ void sample_probe(
    const vec3& origin,
    const ivec2& resolution,
    const vec3& __restrict__ position, 
    vec4* __restrict__ in_rgba, 
    float* __restrict__ in_depth,
    vec4& __restrict__ out_rgba, 
    float& __restrict__ out_depth
) {
    vec2 pix_size {1.0f / (float)resolution.x, 1.0f / (float)resolution.y};

    vec3 dir = normalize(position - origin);
    vec2 uv{0.0};
    uv.y = std::acosf(dir.z);
    uv.x = std::asin(dir.y / std::sinf(uv.y));
    uv /= 2.0f * M_PI;
    ivec2 tex_coords {(int)(uv.x / pix_size.x), (int)(uv.y / pix_size.y)};
    // uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    // if (idx % 1000 == 0) printf("%d tex_coord: %d %d\n", idx, tex_coords.x, tex_coords.y);
    out_rgba = in_rgba[tex_coords.y * resolution.x + tex_coords.x];
    out_depth = in_depth[tex_coords.y * resolution.x + tex_coords.x];
}

__global__ void sample_probe_kernel(
    uint32_t n_elements,
    vec3 origin,
    ivec2 resolution,
    const vec4* __restrict__ positions, 
    vec4* __restrict__ in_rgba, 
    float* __restrict__ in_depth,
    vec4* __restrict__ out_rgba, 
    float* __restrict__ out_depth
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
    sample_probe(origin, resolution, positions[i], in_rgba, in_depth, out_rgba[i], out_depth[i]);
}

__global__ void advance_pos_nerf_kernel_sphere(
    uint32_t n_elements,
	BoundingBox render_aabb,
	mat4x3 render_aabb_to_local,
    NerfPayload* __restrict__ payloads,
    uint32_t sample_index,
	const uint8_t* __restrict__ density_grid,
	uint32_t max_mip,
	float cone_angle_constant
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

    NerfPayload& payload = payloads[i];
	vec3 origin = payload.origin;
	vec3 dir = payload.dir;
	vec3 idir = vec3(1.0f) / dir;

	float t = advance_n_steps(payload.t, cone_angle_constant, ld_random_val(sample_index, payload.idx * 786433));
	t = if_unoccupied_advance_to_next_occupied_voxel(t, cone_angle_constant, {origin, dir}, idir, density_grid, 0, max_mip, render_aabb, render_aabb_to_local);
	if (t >= MAX_DEPTH()) {
		payload.alive = false;
	} else {
		payload.t = t;
	}

}

__global__ void init_rays_in_sphere_kernel(ivec2 resolution, 
    vec3 origin,
    uint32_t n_steps,
    NerfPayload* __restrict__ payloads,
    vec4* __restrict__ frame_buffer,
    float* __restrict__ depth_buffer
) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= resolution.x || y >= resolution.y) {
		return;
	}

	uint32_t idx = x + resolution.x * y;

	vec2 uv = vec2{(float)x /(float)resolution.x, (float)y / (float)resolution.y} * 2.0f * M_PI;
    vec3 dir = { sin(uv.x) * cos(uv.y), sin(uv.x) * sin(uv.y), cos(uv.x) };

	NerfPayload& payload = payloads[idx];
	payload.max_weight = 0.0f;

	depth_buffer[idx] = MAX_DEPTH();

    // frame_buffer[idx].rgb() = vec3(0.0);
    frame_buffer[idx].rgb() = vec3_to_col(dir);
    // printf("%d test: %f %f %f\n", idx, frame_buffer[idx].r, frame_buffer[idx].g, frame_buffer[idx].b);

	payload.origin = origin;
	payload.dir = dir;
	payload.t = 0;
	payload.idx = idx;
	payload.n_steps = n_steps;
	payload.alive = true;
}
    
__global__ void shade_kernel_sphere_nerf(
	const uint32_t n_elements,
	bool gbuffer_hard_edges,
	mat4x3 camera_matrix,
	float depth_scale,
	vec4* __restrict__ rgba,
	float* __restrict__ depth,
	NerfPayload* __restrict__ payloads,
	ERenderMode render_mode,
	bool train_in_linear_colors,
	vec4* __restrict__ frame_buffer,
	float* __restrict__ depth_buffer
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements || render_mode == ERenderMode::Distortion) return;
	NerfPayload& payload = payloads[i];

	vec4 tmp = rgba[i];
	if (render_mode == ERenderMode::Normals) {
		vec3 n = normalize(tmp.xyz());
		tmp.rgb() = (0.5f * n + 0.5f) * tmp.a;
	} else if (render_mode == ERenderMode::Cost) {
		float col = (float)payload.n_steps / 128;
		tmp = {col, col, col, 1.0f};
	} else if (gbuffer_hard_edges && render_mode == ERenderMode::Depth) {
		tmp.rgb() = vec3(depth[i] * depth_scale);
	} else if (gbuffer_hard_edges && render_mode == ERenderMode::Positions) {
		vec3 pos = camera_matrix[3] + payload.dir / dot(payload.dir, camera_matrix[2]) * depth[i];
		tmp.rgb() = (pos - 0.5f) / 2.0f + 0.5f;
	}

	if (!train_in_linear_colors && (render_mode == ERenderMode::Shade || render_mode == ERenderMode::Slice)) {
		// Accumulate in linear colors
		tmp.rgb() = srgb_to_linear(tmp.rgb());
	}

	frame_buffer[payload.idx] = tmp + frame_buffer[payload.idx] * (1.0f - tmp.a);
	if (render_mode != ERenderMode::Slice && tmp.a > 0.2f) {
		depth_buffer[payload.idx] = depth[i];
	}
}

void LightProbe::init_rays_in_sphere(
    const ivec2& resolution,
    const vec3& origin,
    uint32_t spp,
    uint32_t padded_output_width,
    uint32_t n_extra_dims,
    const BoundingBox& render_aabb,
    const mat3& render_aabb_to_local,
    const uint8_t* __restrict__ density_grid,
    uint32_t max_mip,
    float cone_angle_constant,
    uint32_t n_steps,
    cudaStream_t stream,
    vec4* frame_buffer,
    float* depth_buffer
) {
    size_t n_pixels = (size_t)resolution.x * resolution.y;
    enlarge(n_pixels, padded_output_width, n_extra_dims, stream);
    if (m_resolution != resolution) {
        m_resolution = resolution;
        m_render_buffer.resize(m_resolution);
    }
    m_position = origin;

    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
	const dim3 threads = { 16, 8, 1 };
	const dim3 blocks = { div_round_up((uint32_t)resolution.x, threads.x), div_round_up((uint32_t)resolution.y, threads.y), 1 };
    init_rays_in_sphere_kernel<<<threads, blocks, 0, stream>>> (
        resolution,
        m_position,
        n_steps,
        m_rays[0].payload,
        !frame_buffer ? m_render_buffer.frame_buffer() : frame_buffer,
        !depth_buffer ? m_render_buffer.depth_buffer() : depth_buffer
    );
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

	m_n_rays_initialized = resolution.x * resolution.y;

	CUDA_CHECK_THROW(cudaMemsetAsync(m_rays[0].rgba, 0, m_n_rays_initialized * sizeof(vec4), stream));
	CUDA_CHECK_THROW(cudaMemsetAsync(m_rays[0].depth, 0, m_n_rays_initialized * sizeof(float), stream));
}

void LightProbe::shade(
    uint32_t n_hit,
    float depth_scale,
    CudaRenderBufferView& render_buffer,
    cudaStream_t stream
) {
    linear_kernel(shade_kernel_sphere_nerf, 0, stream, 
        n_hit,
        false, // render_gbuffer_hard_edges
        mat4x3(1.0f),
        depth_scale,
        this->rays_hit().rgba,
        this->rays_hit().depth,
        this->rays_hit().payload,
        ERenderMode::Shade,
        true,
        render_buffer.frame_buffer,
        render_buffer.depth_buffer
    );
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
}

}

