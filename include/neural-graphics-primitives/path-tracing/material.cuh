#pragma once

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/common_device.cuh>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_host.h>
#include <tiny-cuda-nn/common_device.h>
#include "common.cuh"
#include <cstdlib>

namespace ngp {
namespace pt {

struct Material;
using ngp::Ray;

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
    __host__ virtual ~Material() {}
    __host__ virtual Material* copy_to_gpu() const = 0;
    __device__ virtual void test() const = 0;
    __device__ virtual bool scatter(
        const Ray& r_in, const HitRecord& rec, vec3& attenuation, Ray& scattered, curandState& rand) const = 0;
};

struct LambertianMaterial : public Material {
  public:
    NGP_HOST_DEVICE LambertianMaterial(const vec3& ks) : albedo(ks) {}
    
    __host__ virtual ~LambertianMaterial() { cudaFree(device); }

    __device__ void test() const override {
      printf("material %f %f %f\n", albedo.r, albedo.g, albedo.b);
    }

    __host__ virtual Material* copy_to_gpu() const override {
      LambertianMaterial *tmp;
      tmp = new LambertianMaterial(albedo);
      memcpy(tmp, this, sizeof(LambertianMaterial));

      CUDA_CHECK_THROW(cudaMalloc((void**)&device, sizeof(LambertianMaterial)));
      CUDA_CHECK_THROW(cudaMemcpy(device, this, sizeof(LambertianMaterial), cudaMemcpyHostToDevice));
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
    LambertianMaterial* device{nullptr};
};

}
}