#pragma once

#include <synerfgine/cuda_helpers.h>
#include <synerfgine/virtual_object.h>

#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/render_buffer.h>

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_host.h>
#include <tiny-cuda-nn/vec.h>

#include <limits>

namespace sng {

using namespace tcnn;
struct BufferSet {
    GPUMemory<vec3> normal;
    GPUMemory<float> depth;
    GPUMemory<size_t> obj_id;
    ivec2 m_resolution;

    ~BufferSet() {
        check_guards();
        normal.free_memory();
        depth.free_memory();
        obj_id.free_memory();
    }
    void check_guards() {
        normal.check_guards();
        depth.check_guards();
        obj_id.check_guards();
    }
    void reset(cudaStream_t stream);
    void resize(const ivec2& resolution);
};


}