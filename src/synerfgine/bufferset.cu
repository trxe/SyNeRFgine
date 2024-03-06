#include <synerfgine/bufferset.cuh>

#include <limits>
#include <vector>

namespace sng {

__global__ void reset_bufferset(uint32_t n_elements, vec3* __restrict__ normals, float* __restrict__ depth, size_t* __restrict__ obj_id);

void BufferSet::reset_vo(cudaStream_t stream) {
    check_guards_vo();
    auto n_elements = m_resolution.x * m_resolution.y;
    triangle_lists.free_memory();
    triangle_bvh_lists.free_memory();
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
}
void BufferSet::reset_frames(cudaStream_t stream) {
    check_guards_frames();
    auto n_elements = m_resolution.x * m_resolution.y;
    linear_kernel(reset_bufferset, 0, stream, n_elements, normal.data(), depth.data(), obj_id.data());
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
}
void BufferSet::set_vos(std::unordered_map<std::string, VirtualObject>& vos) {
    check_guards_vo();
    std::vector<Triangle*> triangles;
    std::vector<TriangleBvhNode*> triangle_bvh_nodes;
    std::vector<Material> materials;
    size_t id = 0;
    for (auto& vo_pair : vos) {
        VirtualObject& vo = vo_pair.second;
        triangles.push_back(vo.gpu_triangles());
        triangle_bvh_nodes.push_back(vo.gpu_triangles_bvh_nodes());
        materials.push_back(vo.get_material());
        vo_names[vo_pair.first] = id++;
    }
    triangle_lists.resize_and_copy_from_host(triangles);
    triangle_bvh_lists.resize_and_copy_from_host(triangle_bvh_nodes);
    triangle_materials.resize_and_copy_from_host(materials);
}

void BufferSet::resize(const ivec2& resolution) {
    m_resolution = resolution;
    auto n_elements = resolution.x * resolution.y;
    check_guards_frames();
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
