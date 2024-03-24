#pragma once
#include <neural-graphics-primitives/testbed.h>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/vec.h>

#include <synerfgine/common.cuh>

namespace sng {

__device__ void sample_probe(
    const vec3& origin,
    const ivec2& resolution,
    const vec3& __restrict__ position, 
    vec4* __restrict__ in_rgba, 
    float* __restrict__ in_depth,
    vec4& __restrict__ out_rgba, 
    float& __restrict__ out_depth
);

struct LightProbeData {
    vec3 position;
    ivec2 resolution;
    vec4* rgba;
    float* depth;
};

class LightProbe : public ngp::Testbed::NerfTracer {
public:
    LightProbe() : 
        m_rgba_texture(std::make_shared<GLTexture>()), 
        m_depth_texture(std::make_shared<GLTexture>()) 
    {
    }
    void init_rays_in_sphere(
        const ivec2& resolution,
        const vec3& origin,
        uint32_t spp,
        uint32_t padded_output_width,
        uint32_t n_extra_dims,
        const vec2& focal_length,
        const BoundingBox& render_aabb,
        const mat3& render_aabb_to_local,
        vec4* frame_buffer,
        float* depth_buffer,
        const uint8_t* __restrict__ density_grid,
        uint32_t max_mip,
        float cone_angle_constant,
        cudaStream_t stream
    );

    void shade(
        uint32_t n_hit,
        float depth_scale,
        const mat4x3& camera_matrix,
        CudaRenderBufferView& render_buffer,
        cudaStream_t stream
    );

    LightProbeData data() {
        return {
            m_position,
            m_resolution,
            m_render_buffer.frame_buffer(),
            m_render_buffer.depth_buffer()
        };
    }

    void sample_colors(const vec3* positions, vec3* out_rgba, float* out_depth);

    vec3 m_position;
    ivec2 m_resolution;
    std::shared_ptr<GLTexture> m_rgba_texture;
    std::shared_ptr<GLTexture> m_depth_texture;
    CudaRenderBuffer m_render_buffer {m_rgba_texture, m_depth_texture};
};

}