#pragma once
#include <filesystem/path.h>
#include "material.cuh"
#include "hittable.cuh"
#include "camera.cuh"
#include <fstream>
#include <vector>
#include <stdexcept>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_host.h>
#include <curand_kernel.h>

// #define TINYOBJLOADER_IMPLEMENTATION
#include <tinyobjloader/tiny_obj_loader.h>
#include <fmt/format.h>
#include <json/json.hpp>
#include <tinylogger/tinylogger.h>
#include <neural-graphics-primitives/path-tracing/hittable.cuh>
#include <neural-graphics-primitives/path-tracing/hittable_list.cuh>
#include <neural-graphics-primitives/render_buffer.h>
#include <neural-graphics-primitives/bounding_box.cuh>

// #define DEBUG_BVH 

namespace ngp {
namespace pt {

namespace fs = filesystem;

__global__ void pt_debug_mat(Material* d_mat);
__global__ void pt_copy_mats(uint32_t n_elements, Material*** nested_mat_list, Material** mat_list);
__global__ void pt_copy_hittables(uint32_t n_elements, Hittable*** nested_hit_list, Hittable** hit_list);
__global__ void pt_free_mats(uint32_t n_elements, Material** mat_list);
__global__ void pt_free_hittables(uint32_t n_elements, Hittable** hit_list);
__global__ void init_rays_world_kernel_nerf(
	uint32_t sample_index,
	ivec2 resolution,
	vec2 focal_length,
	mat4x3 camera_matrix0,
	mat4x3 camera_matrix1,
	vec4 rolling_shutter,
	vec2 screen_center,
	vec3 parallax_shift,
    ngp::BoundingBox render_aabb,
    mat3 render_aabb_to_local,
	bool snap_to_pixel_centers,
	float near_distance,
	float plane_z,
	float aperture_size,
	Foveation foveation,
	Lens lens,
	Buffer2DView<const uint8_t> hidden_area_mask,
	Buffer2DView<const vec2> distortion,
    Ray* __restrict__ rays,
    bool* __restrict__ ends
);

struct World {
    // Uniforms
    ivec2 w_resolution;
    mat3 w_nerf_rot;
    mat3 w_nerf_trans;

    // Per pixel buffers
    curandState* w_pixel_rand_state_gpu;
    vec3* px_beta;
    vec3* px_attenuation;
    vec3* px_aggregation;
    vec3* px_pos;
    float* px_shadow;
    Ray* px_ray;
    Ray* px_prev_ray;
    bool *px_end;
    bool *px_prev_end;
    mat4x3* px_camera;
    HitRecord* px_hit_record;

    Camera w_cam;
    Material** w_material_gpu;
    Hittable** w_hittable_gpu;
    std::vector<std::shared_ptr<Material>> w_materials;
    std::vector<std::shared_ptr<Hittable>> w_hittables;
    std::vector<std::vector<std::shared_ptr<Hittable>>> w_primitives;

    World() : is_active(false) {}
    World(const World& other) = delete;
    World& operator=(const World& other) = delete;
    World(unsigned int resx, unsigned int resy, const std::string& config_fp) : is_active(true), w_resolution(resx, resy) {
        alloc_buffers(resx, resy);
        fs::path fp {config_fp};
        if (!fp.exists()) {
            throw std::runtime_error(fmt::format("Virtual World Configuration at {} does not exist", config_fp));
        }
        std::ifstream f{config_fp};
        nlohmann::json config = nlohmann::json::parse(f);
        // nlohmann::json& cam_conf = config["camera"];
        // init_cam(cam_conf);
        nlohmann::json& mat_conf = config["materials"];
        init_mat(mat_conf);
        nlohmann::json& obj_conf = config["objfile"];
        init_objs(obj_conf);
    }
    ~World() { 
        release(); 
        release_mesh();
    }

    void resize(unsigned int resx, unsigned int resy) {
        if (!is_active) return;
        if (w_resolution.x == resx && w_resolution.y == resy) return;
        w_resolution.x = (int)resx;
        w_resolution.y = (int)resy;
        release();
        alloc_buffers(resx, resy);
    }

    void alloc_buffers(unsigned int resx, unsigned int resy) {
        if (!is_active) return;
        uint32_t n_elements = resx * resy;
        CUDA_CHECK_THROW(cudaMalloc(&w_pixel_rand_state_gpu, n_elements * sizeof(curandState)));
        cudaStream_t stream;
        CUDA_CHECK_THROW(cudaStreamCreate(&stream));
        rand_init_pixels<<<resx, resy, 0, stream>>>(resx, resy, w_pixel_rand_state_gpu);
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

        // CUDA_CHECK_THROW(cudaMalloc(&px_beta, n_elements * sizeof(vec3)));
        // CUDA_CHECK_THROW(cudaMalloc(&px_attenuation, n_elements * sizeof(vec3)));
        // CUDA_CHECK_THROW(cudaMalloc(&px_aggregation, n_elements * sizeof(vec3)));
        // CUDA_CHECK_THROW(cudaMalloc(&px_pos, n_elements * sizeof(vec3)));
        // CUDA_CHECK_THROW(cudaMalloc(&px_shadow, n_elements * sizeof(float)));
        CUDA_CHECK_THROW(cudaMalloc(&px_ray, n_elements * sizeof(Ray)));
        CUDA_CHECK_THROW(cudaMalloc(&px_prev_ray, n_elements * sizeof(Ray)));
        CUDA_CHECK_THROW(cudaMalloc(&px_end, n_elements * sizeof(bool)));
        CUDA_CHECK_THROW(cudaMalloc(&px_prev_end, n_elements * sizeof(bool)));
        CUDA_CHECK_THROW(cudaMalloc(&px_camera, n_elements * sizeof(mat4x3)));
        CUDA_CHECK_THROW(cudaMalloc(&px_hit_record, n_elements * sizeof(HitRecord)));
    }

    void init_rays(
		cudaStream_t stream,
        const uint32_t& sample_index,
        const ivec2& resolution,
        const vec2& focal_length,
        const mat4x3& camera_matrix0,
        const mat4x3& camera_matrix1,
        const vec4& rolling_shutter,
        const vec2& screen_center,
        const vec3& parallax_shift,
        const ngp::BoundingBox& render_aabb,
        const mat3& render_aabb_to_local,
        const bool& snap_to_pixel_centers,
        const float& near_distance,
        const float& plane_z,
        const float& aperture_size,
        const Foveation& foveation,
        const Lens& lens,
        Buffer2DView<const uint8_t> hidden_area_mask,
        Buffer2DView<const vec2> distortion
    );

    void render(
        cudaStream_t stream,
        const CudaRenderBufferView& render_buffer,
        const vec2& focal_length,
        const mat4x3& camera_matrix0,
        const mat4x3& camera_matrix1,
        const vec4& rolling_shutter,
        const vec2& screen_center,
        const Foveation& foveation,
        int visualized_dimension
    );

private:
	INIT_BENCHMARK();

    bool is_active;
    void init_cam(const nlohmann::json& config) {
        auto& lookfrom = config["lookfrom"];
        auto& lookat = config["lookat"];
        auto& vup = config["vup"];
        w_cam.eye = { lookfrom[0].get<float>(), lookfrom[1].get<float>(), lookfrom[2].get<float>() };
        w_cam.at = { lookat[0].get<float>(), lookat[1].get<float>(), lookat[2].get<float>() };
        w_cam.up = { vup[0].get<float>(), vup[1].get<float>(), vup[2].get<float>() };
        w_cam.fov = config["vfov"].get<float>();
        w_cam.aperture = config["aperture"].get<float>();
        w_cam.focus_dist = config["focus_dist"].get<float>();
    }

    void init_mat(const nlohmann::json& all_config) {
        Material*** h_material_list;
        CUDA_CHECK_THROW(cudaMallocManaged(&h_material_list, all_config.size() * sizeof(Material**)));
        size_t id = 0;
        for (auto& config : all_config) {
            std::string type_str {config["type"].get<std::string>()}; 
            std::shared_ptr<Material> mat;
            if (type_str == "lambertian") {
                auto& a = config["albedo"];
                vec3 albedo { a[0].get<float>(), a[1].get<float>(), a[2].get<float>() };
                w_materials.emplace_back(std::make_shared<LambertianMaterial>(albedo));
                Material** d_mat = w_materials.back()->copy_to_gpu();
                h_material_list[id] = d_mat;
            } else {
                throw std::runtime_error(fmt::format("Material type {} not supported", type_str));
            }
            ++id;
        }
        CUDA_CHECK_THROW(cudaMalloc(&w_material_gpu, all_config.size() * sizeof(Material*)));
        int n_element = all_config.size();
        pt_copy_mats<<<n_element, 1>>>(n_element, h_material_list, w_material_gpu);
        CUDA_CHECK_THROW(cudaDeviceSynchronize());
        CUDA_CHECK_THROW(cudaFree(h_material_list));
    }

    void init_objs(const nlohmann::json& all_config) {
        for (auto& config : all_config) {
            std::string obj_fp {config["dir"].get<std::string>()};
            uint32_t material_idx {config["material"].get<uint32_t>()};

            fs::path fp {obj_fp};
            if (!fp.exists()) {
                throw std::runtime_error(fmt::format("Object {} does not exist", obj_fp));
            }

            tinyobj::attrib_t attrib;
            std::vector<tinyobj::shape_t> shapes;
            std::vector<tinyobj::material_t> materials;
            std::string warn, err;

            std::ifstream f{obj_fp, std::ios::in | std::ios::binary};
            if (f.fail()) {
                tlog::error() << "File not found: " << obj_fp;
            }
            bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, &f);

            if (!warn.empty()) {
                tlog::warning() << warn << " while loading '" << obj_fp << "'";
            }

            if (!err.empty()) {
                throw std::runtime_error(fmt::format("Error loading world from {}: {}", obj_fp, err));
            }

            tlog::success() << "Loaded mesh \"" << obj_fp << "\" file with " << shapes.size() << " shapes.";

            Hittable*** h_hittable_list;
#ifdef DEBUG_BVH 
            CUDA_CHECK_THROW(cudaMallocManaged((void**)&h_hittable_list, shapes.size() * sizeof(Hittable*)));
#endif
            size_t shape_id = 0;
            for (auto& shape : shapes) {
                // vec3 center{};
                // uint32_t tri_count{0};
                std::vector<std::shared_ptr<Hittable>> triangles_cpu;
                auto& idxs = shape.mesh.indices;
                auto& verts = attrib.vertices;

                auto get_vec = [verts=verts, idxs=idxs](size_t i) {
                    return vec3(
                        verts[idxs[i].vertex_index * 3], 
                        verts[idxs[i].vertex_index * 3 + 1], 
                        verts[idxs[i].vertex_index * 3 + 2]
                    );
                };
                for (size_t i = 0; i < shape.mesh.indices.size(); i += 3) {
                    triangles_cpu.emplace_back(std::make_shared<Tri>(
                        get_vec(i), get_vec(i+1), get_vec(i+2), material_idx
                    ));
#ifndef DEBUG_BVH 
                    w_hittables.emplace_back(triangles_cpu.back());
#endif
                }
#ifdef DEBUG_BVH 
                w_hittables.emplace_back(std::make_shared<BvhCpu>(triangles_cpu, 0, triangles_cpu.size()));
                Hittable** bvh_gpu_ptr = w_hittables.back()->copy_to_gpu();
                h_hittable_list[shape_id] = bvh_gpu_ptr;
                ++shape_id;
#endif
            }
#ifndef DEBUG_BVH 
            CUDA_CHECK_THROW(cudaMallocManaged((void**)&h_hittable_list, w_hittables.size() * sizeof(Hittable*)));
            for (size_t t= 0; t < w_hittables.size(); ++t) {
                h_hittable_list[t] = w_hittables[t]->copy_to_gpu();
            }
#endif
            int n_element = w_hittables.size();
            CUDA_CHECK_THROW(cudaMalloc(&w_hittable_gpu, n_element * sizeof(Hittable*)));
            pt_copy_hittables<<<n_element, 1>>>(n_element, h_hittable_list, w_hittable_gpu);
            CUDA_CHECK_THROW(cudaDeviceSynchronize());
            CUDA_CHECK_THROW(cudaFree(h_hittable_list));
        };
    }

    void release() {
        // cudaFree(w_material_gpu);
        // cudaFree(w_hittable_gpu);
        cudaFree(w_pixel_rand_state_gpu);
        // cudaFree(px_beta);
        // cudaFree(px_attenuation);
        // cudaFree(px_aggregation);
        // cudaFree(px_pos);
        // cudaFree(px_shadow);
        cudaFree(px_ray);
        cudaFree(px_prev_ray);
        cudaFree(px_end);
        cudaFree(px_prev_end);
        cudaFree(px_camera);
        cudaFree(px_hit_record);
    }

    void release_mesh() {
        auto mat_count = w_materials.size();
        auto hit_count = w_hittables.size();
        pt_free_mats<<<mat_count, 1>>>(mat_count, w_material_gpu);
        pt_free_hittables<<<mat_count, 1>>>(hit_count, w_hittable_gpu);
        cudaDeviceSynchronize();
        cudaFree(w_material_gpu);
        cudaFree(w_hittable_gpu);
    }
};

}
}