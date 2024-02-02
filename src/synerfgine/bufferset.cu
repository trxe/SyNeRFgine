#include <synerfgine/bufferset.cuh>

namespace sng {

__global__ void reset_bufferset(uint32_t n_elements, vec3* __restrict__ normals, float* __restrict__ depth, size_t* __restrict__ obj_id);

void BufferSet::reset(cudaStream_t stream) {
    check_guards();
    auto n_elements = m_resolution.x * m_resolution.y;
    linear_kernel(reset_bufferset, 0, stream, n_elements, normal.data(), depth.data(), obj_id.data());
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
}
void BufferSet::resize(const ivec2& resolution) {
    m_resolution = resolution;
    auto n_elements = resolution.x * resolution.y;
    check_guards();
    normal.resize(n_elements);
    depth.resize(n_elements);
    obj_id.resize(n_elements);
}

__global__ void reset_bufferset(uint32_t n_elements, vec3* __restrict__ normals, float* __restrict__ depth, size_t* __restrict__ obj_id) {
    uint32_t i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= n_elements) {
        return;
    }
    normals[i] = vec3(0.0);
    depth[i] = std::numeric_limits<float>::max();
    obj_id[i] = std::numeric_limits<size_t>::max();
}

} // namespace sng
