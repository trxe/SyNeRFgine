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
	Origin,
	Direction,
	Normal,
	// MaterialIndex,
	// Alive,
};

static const char * img_buffer_type_names[] = {
	"Final",
	"Origin",
	"Direction",
	"Normal",
	// "MaterialIndex",
	// "Alive"
};

struct ObjectTransform {
	NGP_HOST_DEVICE ObjectTransform(TriangleBvhNode* g_node, Triangle* g_tris, const mat3& rot, const vec3& pos, const float& scale) :
		g_node(g_node), g_tris(g_tris), rot(rot), pos(pos), scale(scale) {}
	TriangleBvhNode* g_node;
	Triangle* g_tris;
	mat3 rot;
	vec3 pos;
	float scale;
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

		void enlarge(const ivec2& resolution);
		RaysSoa& rays_hit() { return m_rays_hit; }
		RaysSoa& rays_init() { return m_rays[0]; }
		uint32_t n_rays_initialized() const { return m_n_rays_initialized; }
		ivec2 resolution() const { return m_render_buffer.out_resolution(); }
		void render(
			std::vector<Material>& h_materials, 
			std::vector<VirtualObject>& h_vo, 
			std::vector<Light>& h_light, 
			const Testbed::View& view, 
			const vec2& screen_center,
			uint32_t sample_index,
			const vec2& focal_length,
			bool snap_to_pixel_centers,
			const uint8_t* density_grid_bitfield
		);

        void load(std::vector<vec4>& frame_cpu, std::vector<float>& depth_cpu);

		void sync() {
			CUDA_CHECK_THROW(cudaStreamSynchronize(m_stream_ray));
		}

		void set_objs(std::vector<VirtualObject>& vos) {
			h_world.clear();
			for (auto& obj : vos) {
				h_world.emplace_back(obj.gpu_node(), obj.gpu_triangles(), obj.get_rotate(), obj.get_translate(), obj.get_scale());
			}
			d_world.check_guards();
			d_world.resize_and_copy_from_host(h_world);
		}

		CudaRenderBuffer& render_buffer() { return m_render_buffer; }

		void imgui();

        std::shared_ptr<GLTexture> m_rgba_texture;
        std::shared_ptr<GLTexture> m_depth_texture;
		cudaStream_t m_stream_ray;

	private:
		vec3* buffer_selector(RaysSoa& rays, ImgBufferType to_show) {
			switch (to_show) {
			case ImgBufferType::Final: 
				return rays.rgb;
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

		void set_gpu_materials_light(std::vector<Material>& materials, std::vector<Light>& lights) {
			{
				bool is_dirty = false;
				for (auto& m : materials) {
					is_dirty = is_dirty || m.is_dirty;
					m.is_dirty = false;
				}
				if (is_dirty) {
					d_materials.check_guards();
					d_materials.resize_and_copy_from_host(materials);
				}
			}
			{
				bool is_dirty = false;
				for (auto& l : lights) {
					is_dirty = is_dirty || l.is_dirty;
					l.is_dirty = false;
				}
				if (is_dirty) {
					d_lights.check_guards();
					d_lights.resize_and_copy_from_host(lights);
				}
			}
		}

		RaysSoa m_rays[2];
		RaysSoa m_rays_hit;
        CudaRenderBuffer m_render_buffer {m_rgba_texture, m_depth_texture};
		uint32_t* m_hit_counter;
		uint32_t* m_alive_counter;
		uint32_t m_n_rays_initialized = 0;
		GPUMemoryArena::Allocation m_scratch_alloc;

		GPUMemory<Material> d_materials;
		GPUMemory<Light> d_lights;
		std::vector<ObjectTransform> h_world;
		GPUMemory<ObjectTransform> d_world;
		ImgBufferType m_buffer_to_show{ImgBufferType::Final};

		bool m_view_nerf_shadow{true};
		int m_n_steps{20};
};

}