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
#include <synerfgine/material.h>
#include <tiny-cuda-nn/multi_stream.h>
#include <tiny-cuda-nn/vec.h>

namespace sng {

using namespace tcnn;
namespace fs = std::filesystem;
using ngp::Triangle;
using ngp::TriangleBvh;
using ngp::TriangleBvhNode;

constexpr float MIN_DIST = -5.0;
constexpr float MAX_DIST = 5.0;

static char virtual_object_fp[1024] = "../data/obj/smallbox.obj";

class VirtualObject {
public:
    VirtualObject(const char* fp, const std::string& name);
    ~VirtualObject();
    VirtualObject(const VirtualObject&) = delete;
    VirtualObject& operator=(const VirtualObject&) = delete;
    bool update_triangles(cudaStream_t stream);
    mat4 get_transform();
    vec3 get_center();
    Triangle* gpu_triangles();
    TriangleBvhNode* gpu_triangles_bvh_nodes();
    const Material& get_material() { return vo_material; }
    const std::vector<Triangle>& cpu_triangles();
    void imgui();
    const std::string& get_name() { return name; }

	std::unique_ptr<TriangleBvh> triangles_bvh;

private:
    bool needs_update{true};
    std::string name;
    fs::path file_path;
    vec3 pos;
    vec3 rot;
    float scale{0.50f};
    vec3 center;
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
    Material vo_material;
    
    std::vector<Triangle> triangles_cpu;
	GPUMemory<Triangle> orig_triangles_gpu;
	GPUMemory<Triangle> triangles_gpu;
};

}