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

#include <imgui/imgui.h>
#include <imguizmo/ImGuizmo.h>

#include <synerfgine/common.cuh>

namespace sng {

constexpr float MIN_DIST = -5.0;
constexpr float MAX_DIST = 5.0;

class VirtualObject {
public:
    VirtualObject(uint32_t id, const nlohmann::json& config);
    ~VirtualObject() { triangles_gpu.free_memory(); }
    const mat3& get_rotate() const { return rot; }
    const vec3& get_translate() const { return pos; }
    float get_scale() const { return scale; }
    int32_t get_mat_idx() const { return static_cast<int32_t>(material_id); }
    Triangle* gpu_triangles() { return triangles_gpu.data(); }
    std::shared_ptr<TriangleBvh> bvh() { return triangles_bvh; }
    void imgui();

private:
    std::string name;
    fs::path file_path;
    vec3 pos;
    mat3 rot;
    float scale{1.0f};
    
    std::vector<Triangle> triangles_cpu;
	GPUMemory<Triangle> triangles_gpu;
    std::shared_ptr<ngp::TriangleBvh> triangles_bvh;
    uint32_t material_id;
    uint32_t id;
};

}