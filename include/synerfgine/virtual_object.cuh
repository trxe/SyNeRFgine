#pragma once

#include <chrono>
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
#include <synerfgine/probe.cuh>

namespace sng {

constexpr float MIN_DIST = -5.0;
constexpr float MAX_DIST = 5.0;

enum MovementType {
    Circle,
    Line,
};

struct MovementInfo {
    float c_radius{1.0f};
    vec3 c_center{0.0f};
};

class VirtualObject {
public:
    VirtualObject(uint32_t id, const nlohmann::json& config);
    ~VirtualObject() { triangles_gpu.free_memory(); }
    uint32_t get_id() { return id; }
    const mat3& get_rotate() const { return rot; }
    const vec3& get_translate() const { return pos; }
    float get_scale() const { return scale; }

    mat3& get_rotate_mut() { return rot; }
    vec3& get_translate_mut() { return pos; }
    float& get_scale_mut() { return scale; }
    void next_frame(const float& speed) {
        if (anim_rot_angle == 0.0f) return;
        vec3& ax = anim_rot_axis;
		float cost = std::cos(anim_rot_angle * speed);
		float sint = std::sin(anim_rot_angle * speed);
		anim_next_rot = mat3{
			cost + ax.x * ax.x * (1.0f - cost),        ax.x * ax.y * (1.0f - cost) - ax.z * sint, ax.x * ax.z * (1.0f - cost) + ax.y * sint, 
			ax.x * ax.y * (1.0f - cost) + ax.z * sint, cost + ax.y * ax.y * (1.0f - cost),        ax.y * ax.z * (1.0f - cost) - ax.x * sint, 
			ax.z * ax.y * (1.0f - cost) - ax.y * sint, ax.z * ax.y * (1.0f - cost) + ax.x * sint, cost + ax.z * ax.z * (1.0f - cost) 
		};
        pos = anim_next_rot * (rot * (pos - anim_rot_centre)) + anim_rot_centre;
    }

    int32_t get_mat_idx() const { return material_id; }
    TriangleBvhNode* gpu_node()  { return triangles_bvh->nodes_gpu(); }
    Triangle* gpu_triangles() { return triangles_gpu.data(); }
    std::shared_ptr<TriangleBvh> bvh() { return triangles_bvh; }
    void imgui();

    bool is_dirty{true};

private:
    std::string name;
    fs::path file_path;
    vec3 pos;
    mat3 rot;
    float scale{1.0f};
    
    std::vector<Triangle> triangles_cpu;
	GPUMemory<Triangle> triangles_gpu;
    std::shared_ptr<ngp::TriangleBvh> triangles_bvh;
    int32_t material_id;
    uint32_t id;

    float anim_rot_angle{0.0f};
    mat3 anim_next_rot{mat3::identity()};
    vec3 anim_rot_axis{0.0f, 1.0f, 0.0f};
    vec3 anim_rot_centre{0.0f};
};

}