#pragma once

#include <fstream>
#include <filesystem>
#include <string>
#include <memory>
#include <vector>

// #define TINYOBJLOADER_IMPLEMENTATION
#include <tinyobjloader/tiny_obj_loader.h>

#include <neural-graphics-primitives/triangle.cuh>
#include <neural-graphics-primitives/triangle_bvh.cuh>

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/vec.h>

#include <synerfgine/common.cuh>

namespace sng {

constexpr float MIN_DIST = -5.0;
constexpr float MAX_DIST = 5.0;

class VirtualObject {
public:
    VirtualObject(uint32_t id, const nlohmann::json& config);
    ~VirtualObject();
    mat4 get_transform();
    Triangle* gpu_triangles();
    std::shared_ptr<TriangleBvh> bvh();
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
    
    std::vector<Triangle> triangles_cpu;
	GPUMemory<Triangle> orig_triangles_gpu;
	GPUMemory<Triangle> triangles_gpu;
};

VirtualObject load_virtual_obj(const char* fp, const std::string& name);
// void reset_final_views(size_t n_views, std::vector<RTView>& rt_views, ivec2 resolution);

}