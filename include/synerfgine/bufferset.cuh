#pragma once

#include <synerfgine/cuda_helpers.h>
#include <synerfgine/virtual_object.h>

#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/render_buffer.h>
#include <neural-graphics-primitives/triangle.cuh>
#include <neural-graphics-primitives/triangle_bvh.cuh>

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_host.h>
#include <tiny-cuda-nn/vec.h>

#include <unordered_map>

namespace sng {

using namespace tcnn;
struct BufferSet {
    GPUMemory<vec3> normal;
    GPUMemory<float> depth;
    GPUMemory<size_t> obj_id;
    GPUMemory<Triangle*> triangle_lists;
    GPUMemory<TriangleBvhNode*> triangle_bvh_lists;
    GPUMemory<Material> triangle_materials;
    std::unordered_map<std::string, size_t> vo_names; // corresponds to obj_id
    ivec2 m_resolution;

    vec3* gpu_normals() { return normal.data(); }
    float* gpu_depths() { return depth.data(); }
    size_t* gpu_obj_ids() { return obj_id.data(); }
    Triangle** gpu_objs() { return triangle_lists.data(); }
    TriangleBvhNode** gpu_bvhs() { return triangle_bvh_lists.data(); }
    Material* gpu_materials() { return triangle_materials.data(); }
    size_t vo_count() { return vo_names.size(); }
    size_t get_obj_id(const std::string& name) {
        return vo_names.count(name) ? vo_names[name] : std::numeric_limits<size_t>::max();
    }

    ~BufferSet() {
        check_guards_frames();
        normal.free_memory();
        depth.free_memory();
        obj_id.free_memory();
        check_guards_vo();
        triangle_lists.free_memory();
        triangle_bvh_lists.free_memory();
        triangle_materials.free_memory();
    }
    void check_guards_frames() {
        normal.check_guards();
        depth.check_guards();
        obj_id.check_guards();
    }
    void check_guards_vo() {
        triangle_lists.check_guards();
        triangle_bvh_lists.check_guards();
        triangle_materials.check_guards();
    }
    void reset_frames(cudaStream_t stream);
    void reset_vo(cudaStream_t stream);
    void resize(const ivec2& resolution);
    void set_vos(std::unordered_map<std::string, VirtualObject>& vos);
    void shade(cudaStream_t stream);
};


}