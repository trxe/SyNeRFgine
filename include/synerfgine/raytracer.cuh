#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/gpu_memory.h>

#include <neural-graphics-primitives/bounding_box.cuh>
#include <neural-graphics-primitives/nerf.h>
#include <neural-graphics-primitives/render_buffer.h>
#include <neural-graphics-primitives/testbed.h>
#include <synerfgine/common.cuh>
#include <synerfgine/material.cuh>
#include <synerfgine/virtual_object.cuh>
#include <synerfgine/light.cuh>

#include <cstdint>

namespace sng {

enum ImgBufferType {
	Final,
	NextOrigin,
	SrcOrigin,
	NextDirection,
	SrcDirection,
	Normal,
	Depth,
	// MaterialIndex,
	// Alive,
};

static const char * img_buffer_type_names[] = {
	"Final",
	"Next Origin",
	"Src Origin",
	"Next Direction",
	"Src Direction",
	"Normal",
	"Depth",
	// "MaterialIndex",
	// "Alive"
};

enum ImgFilterType {
	Original,
	Bilateral,
};

static const char * img_filter_type_names[] = {
	"Original",
	"Bilateral",
};

__device__ vec4 shade_object(const vec3& wi, SampledRay& ray, const uint32_t& shadow_count, HitRecord& hit_info,
	const Light* __restrict__ lights, const size_t& light_count, 
	const ObjectTransform* __restrict__ objects, const size_t& object_count, 
	const Material* __restrict__ materials, const size_t& material_count, 
	const uint32_t& n_steps, const float& cone_angle_constant, 
	const uint8_t* __restrict__ density_grid, const uint32_t& min_mip, const uint32_t& max_mip, 
	const BoundingBox& render_aabb, const mat3& render_aabb_to_local,
	curandState_t& rand_state, bool no_shadow);

struct RaysSoa {
#if defined(__CUDACC__) || (defined(__clang__) && defined(__CUDA__))
	void copy_from_other_async(const RaysSoa& other, cudaStream_t stream) {
		CUDA_CHECK_THROW(cudaMemcpyAsync(origin, other.origin, size * sizeof(vec3), cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK_THROW(cudaMemcpyAsync(dir, other.dir, size * sizeof(float), cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK_THROW(cudaMemcpyAsync(rgba, other.rgba, size * sizeof(vec4), cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK_THROW(cudaMemcpyAsync(depth, other.depth, size * sizeof(float), cudaMemcpyDeviceToDevice, stream));
	}
#endif

	void set(vec3* origin, vec3* dir, vec4* rgba, float* depth, size_t size) {
		this->origin = origin;
		this->dir = dir;
		this->rgba = rgba;
		this->depth = depth;
		this->size = size;
	}

	vec3* origin;
	vec3* dir;
	vec4* rgba;
	float* depth;
	size_t size;
};

class RayTracer {
	public:
		RayTracer() : m_rgba_texture(std::make_shared<GLTexture>()), m_depth_texture(std::make_shared<GLTexture>()) {
			CUDA_CHECK_THROW(cudaStreamCreate(&m_stream_ray));
		}

		void init_rays_from_camera(
			uint32_t sample_index,
			const vec2& focal_length,
			const mat4x3& camera,
			const vec2& screen_center,
			bool snap_to_pixel_centers
		);

		int filter_type() const { return static_cast<int>(m_filter_to_use); }
		void enlarge(const ivec2& resolution);
		// RaysSoa& rays_hit() { return m_rays_hit; }
		RaysSoa& rays_init() { return m_rays[0]; }
		uint32_t n_rays_initialized() const { return m_n_rays_initialized; }
		void reset_accumulation() { m_reset_color_buffer = true; }
		bool needs_reset() const { return m_reset_color_buffer; }
		ivec2 resolution() const { 
			return m_render_buffer.out_resolution(); 
		}
		vec4* get_tmp_frame_buffer() { return m_rays[0].rgba; }
		float* get_tmp_depth_buffer() { return m_rays[0].depth; }
		void render(
			std::vector<VirtualObject>& h_vo, 
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
		);

		void overlay(CudaRenderBufferView nerf_scene, size_t syn_px_scale, 
			ngp::EColorSpace color_space, ngp::ETonemapCurve tonemap_curve, float exposure);
        void load(std::vector<vec4>& frame_cpu, std::vector<float>& depth_cpu);

		void sync() {
			CUDA_CHECK_THROW(cudaStreamSynchronize(m_stream_ray));
		}

		CudaRenderBuffer& render_buffer() { return m_render_buffer; }

		void imgui();

        std::shared_ptr<GLTexture> m_rgba_texture;
        std::shared_ptr<GLTexture> m_depth_texture;
		cudaStream_t m_stream_ray;

		bool m_view_nerf_shadow{true};
		bool m_show_virtual_obj{true};
		int m_n_steps{8};
		int m_samples = 2;
		int m_ray_iters = 2;
		int m_shadow_iters = 4;
		float m_lens_angle_constant = 0.009f;
		float m_blend_ratio = 0.5f;
		bool m_use_blend_ratio = false;
		float m_attenuation_coeff = 1.0f;

	private:
		RaysSoa m_rays[1];
        CudaRenderBuffer m_render_buffer {m_rgba_texture, m_depth_texture};
		uint32_t* m_hit_counter;
		uint32_t* m_alive_counter;
		curandState_t* m_rand_state;
		uint32_t m_n_rays_initialized = 0;
		GPUMemoryArena::Allocation m_scratch_alloc;
		bool m_reset_color_buffer{true};

		ImgBufferType m_buffer_to_show{ImgBufferType::Final};
		ImgFilterType m_filter_to_use{ImgFilterType::Bilateral};
};

}