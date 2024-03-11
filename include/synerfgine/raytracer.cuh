#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/gpu_memory.h>

#include <neural-graphics-primitives/nerf.h>
#include <neural-graphics-primitives/bounding_box.cuh>
#include <neural-graphics-primitives/render_buffer.h>
#include <synerfgine/common.cuh>

namespace sng {

struct RayPayload {
	vec3 origin;
	vec3 dir;
	float t;
	float max_weight;
	uint32_t idx;
	uint16_t n_steps;
	bool alive;
};

struct RaysSoa {
#if defined(__CUDACC__) || (defined(__clang__) && defined(__CUDA__))
	void copy_from_other_async(const RaysSoa& other, cudaStream_t stream) {
		CUDA_CHECK_THROW(cudaMemcpyAsync(rgba, other.rgba, size * sizeof(vec4), cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK_THROW(cudaMemcpyAsync(depth, other.depth, size * sizeof(float), cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK_THROW(cudaMemcpyAsync(payload, other.payload, size * sizeof(RayPayload), cudaMemcpyDeviceToDevice, stream));
	}
#endif

	void set(vec4* rgba, float* depth, RayPayload* payload, size_t size) {
		this->rgba = rgba;
		this->depth = depth;
		this->payload = payload;
		this->size = size;
	}

	vec4* rgba;
	float* depth;
	RayPayload* payload;
	size_t size;
};

class RayTracer {
	public:
		RayTracer() : m_rgba_texture(std::make_shared<GLTexture>()), m_depth_texture(std::make_shared<GLTexture>()) {
			CUDA_CHECK_THROW(cudaStreamCreate(&m_stream_ray));
		}

		void init_rays_from_camera(
			uint32_t spp,
			uint32_t padded_output_width,
			// uint32_t n_extra_dims,
			const ivec2& resolution,
			const vec2& focal_length,
			const mat4x3& camera_matrix0,
			const mat4x3& camera_matrix1,
			const vec4& rolling_shutter,
			const vec2& screen_center,
			const vec3& parallax_shift,
			bool snap_to_pixel_centers,
			const ngp::BoundingBox& render_aabb,
			const mat3& render_aabb_to_local,
			float near_distance,
			float plane_z,
			float aperture_size,
			const Foveation& foveation,
			const Lens& lens,
			const Buffer2DView<const vec4>& envmap,
			const Buffer2DView<const vec2>& distortion,
			// vec4* frame_buffer,
			// float* depth_buffer,
			// const Buffer2DView<const uint8_t>& hidden_area_mask,
			const uint8_t* grid,
			int show_accel,
			uint32_t max_mip,
			float cone_angle_constant
			// ERenderMode render_mode,
			// , cudaStream_t stream
		);

		uint32_t trace(
			// const std::shared_ptr<NerfNetwork<network_precision_t>>& network,
			const ngp::BoundingBox& render_aabb,
			const mat3& render_aabb_to_local,
			const ngp::BoundingBox& train_aabb,
			const vec2& focal_length,
			float cone_angle_constant,
			const uint8_t* grid,
			ERenderMode render_mode,
			const mat4x3 &camera_matrix,
			float depth_scale,
			int visualized_layer,
			int visualized_dim,
			// ENerfActivation rgb_activation,
			// ENerfActivation density_activation,
			int show_accel,
			uint32_t max_mip,
			float min_transmittance,
			float glow_y_cutoff,
			int glow_mode,
			const float* extra_dims_gpu
			// , cudaStream_t stream
		);

		void enlarge(const ivec2& resolution);
		RaysSoa& rays_hit() { return m_rays_hit; }
		RaysSoa& rays_init() { return m_rays[0]; }
		uint32_t n_rays_initialized() const { return m_n_rays_initialized; }
		const ivec2& resolution() const { return m_render_buffer.out_resolution(); }

        void test();
        void load(std::vector<vec4>& frame_cpu, std::vector<float>& depth_cpu);
		void sync() {
			CUDA_CHECK_THROW(cudaStreamSynchronize(m_stream_ray));
		}

        std::shared_ptr<GLTexture> m_rgba_texture;
        std::shared_ptr<GLTexture> m_depth_texture;
		cudaStream_t m_stream_ray;

	private:
		RaysSoa m_rays[2];
		RaysSoa m_rays_hit;
        CudaRenderBuffer m_render_buffer {m_rgba_texture, m_depth_texture};
		uint32_t* m_hit_counter;
		uint32_t* m_alive_counter;
		uint32_t m_n_rays_initialized = 0;
		GPUMemoryArena::Allocation m_scratch_alloc;

};

}