#pragma once

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/triangle.cuh>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_device.h>
#include "common.cuh"
#include "aabb.cuh"

#include <cstdint>

#include "material.cuh"

namespace ngp { 
namespace pt {

using namespace tcnn;

struct Hittable {
  public:
    __host__ virtual ~Hittable() {}
    NGP_HOST_DEVICE virtual AABB bounding_box() const = 0;
    NGP_HOST_DEVICE virtual vec3 center() const = 0;
    __device__ virtual bool hit(const ngp::Ray& r, float ray_tmin, float ray_tmax, HitRecord& rec) const = 0;
    __host__ virtual Hittable* copy_to_gpu() const = 0;
    __host__ virtual Hittable* copy_to_gpu(const cudaStream_t& stream) const = 0;
    __host__ Hittable* get_device() {
        return device;
    }
protected:
    Hittable* device {nullptr};
};

struct Tri : public Hittable {
public:
    vec3 point[3];
    uint32_t material_idx;

    NGP_HOST_DEVICE Tri(const vec3& p1, const vec3& p2, const vec3& p3, uint32_t material_idx) : material_idx(material_idx) {
        point[0] = p1;
        point[1] = p2;
        point[2] = p3;
    }

    __host__ virtual ~Tri()  { cudaFree(device); }

    NGP_HOST_DEVICE vec3 center() const override {
        return (point[0] + point[1] + point[2]) / 3.0f;
    }

    NGP_HOST_DEVICE AABB bounding_box() const override {
        float xleft = std::min({point[0].x, point[1].x, point[2].x});
        float xright = std::max({point[0].x, point[1].x, point[2].x});
        float yleft = std::min({point[0].y, point[1].y, point[2].y});
        float yright = std::max({point[0].y, point[1].y, point[2].y});
        float zleft = std::min({point[0].z, point[1].z, point[2].z});
        float zright = std::max({point[0].z, point[1].z, point[2].z});
        return {xleft, xright, yleft, yright, zleft, zright};
    }

    __host__ virtual Hittable* copy_to_gpu() const override {
        if (!device) {
            CUDA_CHECK_THROW(cudaMalloc((void**)&device, sizeof(Tri)));
        }
		CUDA_CHECK_THROW(cudaMemcpy(device, this, sizeof(Tri), cudaMemcpyHostToDevice));
		return device;
    }

    __host__ virtual Hittable* copy_to_gpu(const cudaStream_t& stream) const override {
        if (!device) {
            CUDA_CHECK_THROW(cudaMallocAsync((void**)&device, sizeof(Tri), stream));
        }
		CUDA_CHECK_THROW(cudaMemcpyAsync(device, this, sizeof(Tri), cudaMemcpyHostToDevice, stream));
		return device;
    }

    __device__ bool hit(const ngp::Ray& r, float ray_tmin, float ray_tmax, HitRecord& rec) const override {
        // Möller–Trumbore
        vec3 edge1 = point[1] - point[0];
        vec3 edge2 = point[2] - point[0];
        vec3 ray_cross_e2 = cross(r.d, edge2);
        float det = dot(edge1, ray_cross_e2);

        if (det > -PT_EPSILON && det < PT_EPSILON)
            return false;    // This ray is parallel to this triangle.

        float inv_det = 1.0 / det;
        vec3 s = r.o - point[0];
        float u = inv_det * dot(s, ray_cross_e2);

        if (u < 0 || u > 1)
            return false;

        vec3 s_cross_e1 = cross(s, edge1);
        float v = inv_det * dot(r.d, s_cross_e1);

        if (v < 0 || u + v > 1)
            return false;

        // At this stage we can compute t to find out where the intersection point is on the line.
        float t = inv_det * dot(edge2, s_cross_e1);

        if (t > PT_EPSILON) // ray intersection
        {
            rec.t = t;
            rec.pos = r.o + r.d * t;
            rec.material_idx = material_idx;
            rec.normal = normalize(cross(edge1, edge2));
            rec.front_face = true;
            return true;
        }
        else // This means that there is a line intersection but not a ray intersection.
            return false;

    }
};


}
}