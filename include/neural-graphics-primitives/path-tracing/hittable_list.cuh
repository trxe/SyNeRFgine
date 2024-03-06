#pragma once

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/common_device.cuh>
#include "hittable.cuh"
#include "aabb.cuh"
#include <algorithm>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace ngp {
namespace pt {

using ngp::Ray;

class Bvh : public Hittable {
  public:
    __host__ Bvh(std::vector<std::shared_ptr<Hittable>>& objects, vec3 center, size_t start, size_t end) {
        for (auto& obj : objects) {
            obj->copy_to_gpu();
        }

        int axis = std::rand() % 3;
        auto comparator = (axis == 0) ? box_x_compare
                        : (axis == 1) ? box_y_compare
                                      : box_z_compare;

        size_t object_span = end - start;

        if (object_span == 1) {
            left = right = objects[start];
        } else if (object_span == 2) {
            if (comparator(objects[start], objects[start+1])) {
                left = objects[start];
                right = objects[start+1];
            } else {
                left = objects[start+1];
                right = objects[start];
            }
        } else {
            std::sort(objects.begin() + start, objects.begin() + end, comparator);

            auto mid = start + object_span/2;
            left = std::make_shared<Bvh>(objects, center, start, mid);
            right = std::make_shared<Bvh>(objects, center, mid, end);
        }
        d_left = !left ? nullptr : left->copy_to_gpu();
        d_right = !right ? nullptr : right->copy_to_gpu();

        _bounding_box = AABB(left->bounding_box(), right->bounding_box());
    }

    __host__ virtual ~Bvh() {
        left = right = nullptr;
        cudaFree(device);
    }

    __device__ bool hit(const Ray& orig_ray, float ray_tmin, float ray_tmax, HitRecord& rec) const override {
        Ray r {orig_ray.o - _center, orig_ray.d};
        if (!_bounding_box.hit(r, ray_tmin, ray_tmax))
            return false;

        bool hit_left = d_left->hit(r, ray_tmin, ray_tmax, rec);
        bool hit_right = d_right->hit(r, ray_tmin, hit_left ? rec.t : ray_tmax, rec);

        return hit_left || hit_right;
    }

    NGP_HOST_DEVICE vec3 center() const override { return _center; }
    NGP_HOST_DEVICE AABB bounding_box() const override { return _bounding_box; }

    __host__ virtual Hittable* copy_to_gpu() const {
        if (!device) {
            CUDA_CHECK_THROW(cudaMalloc((void**)&device, sizeof(Bvh)));
        }
		CUDA_CHECK_THROW(cudaMemcpy(device, this, sizeof(Bvh), cudaMemcpyHostToDevice));
		return device;
    }

    __host__ virtual Hittable* copy_to_gpu(const cudaStream_t& stream) const {
        if (!device) {
            CUDA_CHECK_THROW(cudaMallocAsync((void**)&device, sizeof(Bvh), stream));
        }
		CUDA_CHECK_THROW(cudaMemcpyAsync(device, this, sizeof(Bvh), cudaMemcpyHostToDevice, stream));
		return device;
    }


  private:
    std::shared_ptr<Hittable> left;
    Hittable* d_left;
    std::shared_ptr<Hittable> right;
    Hittable* d_right;
    vec3 _center;
    AABB _bounding_box;

    __host__ static bool box_compare(
        const std::shared_ptr<Hittable> a, const std::shared_ptr<Hittable> b, int axis_index
    ) {
        return a->bounding_box().axis(axis_index)[0] < b->bounding_box().axis(axis_index)[0];
    }

    __host__ static bool box_x_compare (const std::shared_ptr<Hittable> a, const std::shared_ptr<Hittable> b) {
        return box_compare(a, b, 0);
    }

    __host__ static bool box_y_compare (const std::shared_ptr<Hittable> a, const std::shared_ptr<Hittable> b) {
        return box_compare(a, b, 1);
    }

    __host__ static bool box_z_compare (const std::shared_ptr<Hittable> a, const std::shared_ptr<Hittable> b) {
        return box_compare(a, b, 2);
    }
};

}
}