#pragma once
#include <fstream>
#include <filesystem>
#include <string>
#include <memory>
#include <vector>

#include <tiny-cuda-nn/gpu_memory.h>

// #define TINYOBJLOADER_IMPLEMENTATION
#include <tinyobjloader/tiny_obj_loader.h>

#include <neural-graphics-primitives/triangle.cuh>
#include <neural-graphics-primitives/triangle_bvh.cuh>

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/multi_stream.h>
#include <tiny-cuda-nn/vec.h>

namespace sng {

using namespace tcnn;
namespace fs = std::filesystem;
using ngp::Triangle;
using ngp::TriangleBvh;

constexpr float MIN_DIST = -5.0;
constexpr float MAX_DIST = 5.0;

static char virtual_object_fp[1024] = "../data/obj/smallbox.obj";

struct Material {
    vec3 ka;
    vec3 kd;
    vec3 ks;
    float n;
};

class VirtualObject {
public:
    VirtualObject(const char* fp, const std::string& name);
    ~VirtualObject();
    VirtualObject(const VirtualObject&) = delete;
    VirtualObject& operator=(const VirtualObject&) = delete;
    bool update_triangles(cudaStream_t stream);
    mat4 get_transform();
    Triangle* gpu_triangles();
    const std::vector<Triangle>& cpu_triangles();
    void add_to_vo_list(std::unordered_map<std::string, VirtualObject>&);
    void imgui();

private:
    bool needs_update{true};
    std::string name;
    fs::path file_path;
    vec3 pos;
    vec3 rot;
    float scale{1.0f};
    vec3 center;
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
    Material vo_material;
    
    std::vector<Triangle> triangles_cpu;
	GPUMemory<Triangle> orig_triangles_gpu;
	GPUMemory<Triangle> triangles_gpu;
	std::unique_ptr<TriangleBvh> triangles_bvh;
};

}