#pragma once

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/common_device.cuh>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_host.h>
#include <tiny-cuda-nn/common_device.h>
#include "common.cuh"
#include <cstdlib>
#include <stdio.h>

namespace ngp {
namespace pt {

struct Material;
struct LambertianMaterial;

using ngp::Ray;

__global__ void init_material_lambertian(Material** __restrict__ d_mat, vec3 albedo);

struct HitRecord {
  public:
    vec3 pos;
    vec3 normal;
    double t;
    uint32_t material_idx;
    bool front_face;

    NGP_HOST_DEVICE void set_face_normal(const Ray& r, const vec3& outward_normal) {
        front_face = dot(r.d, outward_normal) < 0.0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

struct Material {
public:
    NGP_HOST_DEVICE virtual ~Material() {}
    NGP_HOST_DEVICE virtual void test() const = 0;
    __host__ virtual Material** copy_to_gpu() const = 0;
    __device__ virtual bool scatter(
        const Ray& r_in, const HitRecord& rec, vec3& attenuation, Ray& scattered, curandState& rand) const = 0;
};

struct LambertianMaterial : public Material {
  public:
    NGP_HOST_DEVICE LambertianMaterial(const vec3& ks) : albedo(ks) {}
    
    NGP_HOST_DEVICE virtual ~LambertianMaterial() {}

    NGP_HOST_DEVICE void test() const override {
      printf("material %f %f %f\n", albedo.r, albedo.g, albedo.b);
    }

    __host__ virtual Material** copy_to_gpu() const override {
      Material** device;
      CUDA_CHECK_THROW(cudaMalloc((void**)&device, sizeof(Material**)));
      init_material_lambertian<<<1,1>>>(device, albedo);
      CUDA_CHECK_THROW(cudaDeviceSynchronize());
      return device;
    }

    __device__ bool scatter(const Ray& r_in, const HitRecord& rec, vec3& attenuation, Ray& scattered, curandState& rand) const override {
        auto scatter_direction = rec.normal + Rand::random_unit_vector(&rand);
        scattered.o = rec.pos;
        scattered.d = scatter_direction;
        attenuation = albedo;
        return true;
    }

  private:
    vec3 albedo;
};

}
}