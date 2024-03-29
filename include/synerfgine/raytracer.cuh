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

struct RaysSoa {
#if defined(__CUDACC__) || (defined(__clang__) && defined(__CUDA__))
	void copy_from_other_async(const RaysSoa& other, cudaStream_t stream) {
		CUDA_CHECK_THROW(cudaMemcpyAsync(rgb, other.rgb, size * sizeof(vec3), cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK_THROW(cudaMemcpyAsync(depth, other.depth, size * sizeof(float), cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK_THROW(cudaMemcpyAsync(origin, other.origin, size * sizeof(vec3), cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK_THROW(cudaMemcpyAsync(dir, other.dir, size * sizeof(vec3), cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK_THROW(cudaMemcpyAsync(normal, other.normal, size * sizeof(vec3), cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK_THROW(cudaMemcpyAsync(mat_idx, other.mat_idx, size * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK_THROW(cudaMemcpyAsync(t, other.t, size * sizeof(float), cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK_THROW(cudaMemcpyAsync(alive, other.alive, size * sizeof(bool), cudaMemcpyDeviceToDevice, stream));
	}
#endif

	void set(vec3* rgb, float* depth, vec3* origin, vec3* dir, vec3* normal, int32_t* mat_idx, float* t, bool* alive, size_t size) {
		this->rgb = rgb;
		this->depth = depth;
		this->origin = origin;
		this->dir = dir;
		this->normal = normal;
		this->mat_idx = mat_idx;
		this->t = t;
		this->alive = alive;
		this->size = size;
	}

	vec3* rgb;
	float* depth;
	vec3* origin;
	vec3* dir;
	vec3* normal;
	int32_t* mat_idx;
	float* t;
	bool* alive;
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
		ivec2 resolution() const { 
			return m_render_buffer.out_resolution(); 
		}
		void render(
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
		);

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
		int m_n_steps{20};
		int m_ray_iters = 1;
		int m_shadow_iters = 4;
		float m_attenuation_coeff = 0.5f;

	private:
		RaysSoa m_rays[2];
		// RaysSoa m_rays_hit;
        CudaRenderBuffer m_render_buffer {m_rgba_texture, m_depth_texture};
		uint32_t* m_hit_counter;
		uint32_t* m_alive_counter;
		curandState_t* m_rand_state;
		uint32_t m_n_rays_initialized = 0;
		GPUMemoryArena::Allocation m_scratch_alloc;

		ImgBufferType m_buffer_to_show{ImgBufferType::Final};
		ImgFilterType m_filter_to_use{ImgFilterType::Bilateral};
};

}