#pragma once

#include <synerfgine/camera.h>
#include <synerfgine/cuda_helpers.h>
#include <synerfgine/light.cuh>
#include <synerfgine/virtual_object.h>

#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/render_buffer.h>

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_host.h>

#include <vector>

namespace sng {

using ngp::CudaRenderBuffer;
using ngp::GLTexture;

class Display;

class SyntheticWorld {
public:
    SyntheticWorld();
    bool handle(CudaDevice& device, const ivec2& resolution);
    void create_object(const std::string& filename);
    // void create_camera(const vec3& eye, const vec3& at, const vec3& up = vec3(0.0, 1.0, 0.0));
    // void camera_position(const vec3& eye);
    // void camera_look_at(const vec3& at);
    std::unordered_map<std::string, VirtualObject>& objects() { return m_objects; }
    void delete_object(const std::string& name) {
        m_objects.erase(name);
    }
    std::shared_ptr<CudaRenderBuffer> render_buffer() { return m_render_buffer; }
    const Camera& camera() { return m_camera; }
    Camera& mut_camera() { return m_camera; }

private:
    friend class sng::Display;
    void draw_object_async(CudaDevice& device, VirtualObject& vo);
    // std::vector<Light> m_lights;
    std::unordered_map<std::string, VirtualObject> m_objects;
    // std::vector<Camera> m_cameras;
    Camera m_camera;

    // Buffers and resolution
	std::shared_ptr<GLTexture> m_rgba_render_textures;
	std::shared_ptr<GLTexture> m_depth_render_textures;
	std::shared_ptr<CudaRenderBuffer> m_render_buffer;
	CudaRenderBufferView m_render_buffer_view;
    ivec2 m_resolution;
};

}