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

#include <cstdint>

namespace sng {

enum ImgBufferType {
	Origin,
	Direction,
	Normal,
	// MaterialIndex,
	// Alive,
};

static const char * img_buffer_type_names[] = {
	"Origin",
	"Direction",
	"Normal",
	// "MaterialIndex",
	// "Alive"
};

struct RaysSoa {
#if defined(__CUDACC__) || (defined(__clang__) && defined(__CUDA__))
	void copy_from_other_async(const RaysSoa& other, cudaStream_t stream) {
		CUDA_CHECK_THROW(cudaMemcpyAsync(rgba, other.rgba, size * sizeof(vec4), cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK_THROW(cudaMemcpyAsync(depth, other.depth, size * sizeof(float), cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK_THROW(cudaMemcpyAsync(depth, other.origin, size * sizeof(vec3), cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK_THROW(cudaMemcpyAsync(depth, other.dir, size * sizeof(vec3), cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK_THROW(cudaMemcpyAsync(depth, other.normal, size * sizeof(vec3), cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK_THROW(cudaMemcpyAsync(depth, other.mat_idx, size * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK_THROW(cudaMemcpyAsync(depth, other.t, size * sizeof(float), cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK_THROW(cudaMemcpyAsync(depth, other.alive, size * sizeof(bool), cudaMemcpyDeviceToDevice, stream));
	}
#endif

	void set(vec4* rgba, float* depth, vec3* origin, vec3* dir, vec3* normal, int32_t* mat_idx, float* t, bool* alive, size_t size) {
		this->rgba = rgba;
		this->depth = depth;
		this->origin = origin;
		this->dir = dir;
		this->normal = normal;
		this->mat_idx = mat_idx;
		this->t = t;
		this->alive = alive;
		this->size = size;
	}

	vec4* rgba;
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

		void enlarge(const ivec2& resolution);
		RaysSoa& rays_hit() { return m_rays_hit; }
		RaysSoa& rays_init() { return m_rays[0]; }
		uint32_t n_rays_initialized() const { return m_n_rays_initialized; }
		ivec2 resolution() const { return m_render_buffer.out_resolution(); }
		void render(
			const std::vector<Material>& h_materials, 
			std::vector<VirtualObject>& h_vo, 
			const Testbed::View& view, 
			const vec2& screen_center,
			uint32_t sample_index,
			const vec2& focal_length,
			bool snap_to_pixel_centers
		);

        void load(std::vector<vec4>& frame_cpu, std::vector<float>& depth_cpu);
		void sync() {
			CUDA_CHECK_THROW(cudaStreamSynchronize(m_stream_ray));
		}

		void imgui();

        std::shared_ptr<GLTexture> m_rgba_texture;
        std::shared_ptr<GLTexture> m_depth_texture;
		cudaStream_t m_stream_ray;

	private:
		vec3* buffer_selector(RaysSoa& rays, ImgBufferType to_show) {
			switch (to_show) {
			case ImgBufferType::Origin: 
				return rays.origin;
			case ImgBufferType::Direction: 
				return rays.dir;
			case ImgBufferType::Normal: 
				return rays.normal;
			default:
				return nullptr;
			}
		}

		RaysSoa m_rays[2];
		RaysSoa m_rays_hit;
        CudaRenderBuffer m_render_buffer {m_rgba_texture, m_depth_texture};
		uint32_t* m_hit_counter;
		uint32_t* m_alive_counter;
		uint32_t m_n_rays_initialized = 0;
		GPUMemoryArena::Allocation m_scratch_alloc;

		// std::vector<Material*> h_material_gpu_ptrs;
		// Material** d_material_gpu_ptrs;
		// std::vector<> h_material_gpu_ptrs;
		// Material** d_material_gpu_ptrs;
		ImgBufferType m_buffer_to_show{ImgBufferType::Normal};

};

}