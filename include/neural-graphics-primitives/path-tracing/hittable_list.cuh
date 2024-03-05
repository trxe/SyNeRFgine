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
#include <stdio.h>

namespace ngp {
namespace pt {

using ngp::Ray;
__global__ void pt_init_bvh(Hittable** hittable, Hittable** left, Hittable** right, vec3 center, AABB bbox);
__global__ void pt_print_bvh(Hittable** bvh);

class BvhCpu : public Hittable {
  public:
    __host__ BvhCpu(std::vector<std::shared_ptr<Hittable>>& objects, size_t start, size_t end) {
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
            left = std::make_shared<BvhCpu>(objects, start, mid);
            right = std::make_shared<BvhCpu>(objects, mid, end);
        }

        _bounding_box = AABB(left->bounding_box(), right->bounding_box());
        _center = (left->center() + left->center()) / 2.0f;
    }

    __host__ ~BvhCpu() {
        left = right = nullptr;
    }

    NGP_HOST_DEVICE void print(uint32_t indent = 0) const override {
        char* str_indent = new char[indent];
        for (uint32_t i = 0; i < indent; ++i) str_indent[i] = ' ';
        printf(">>%sBVH: {%f, %f, %f} -> {%f, %f, %f}\n", str_indent,
            _bounding_box.x[0], _bounding_box.y[0], _bounding_box.z[0],
            _bounding_box.x[1], _bounding_box.y[1], _bounding_box.z[1]);
        left->print(indent + 1);
        right->print(indent + 1);
        delete str_indent;
    }

    __device__ bool hit(const Ray& orig_ray, float ray_tmin, float ray_tmax, HitRecord& rec) const override {
        // noop on device.
        return false;
    }

    NGP_HOST_DEVICE vec3 center() const override { return _center; }
    NGP_HOST_DEVICE AABB bounding_box() const override { return _bounding_box; }

    __host__ virtual Hittable** copy_to_gpu() const override {
        Hittable** device;
        Hittable** d_left = left->copy_to_gpu();
        Hittable** d_right = right->copy_to_gpu();
        CUDA_CHECK_THROW(cudaMalloc((void**)&device, sizeof(Hittable**)));
        pt_init_bvh<<<1,1>>>(device, d_left, d_right, _center, _bounding_box);
        CUDA_CHECK_THROW(cudaDeviceSynchronize());
		return device;
    }


  private:
    std::shared_ptr<Hittable> left;
    std::shared_ptr<Hittable> right;
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

class BvhGpu : public Hittable {
public:
    __device__ BvhGpu(Hittable** left_child, Hittable** right_child, vec3 center, AABB bbox) {
        left = *left_child;
        right = *right_child;
        _center = center;
        _bounding_box = bbox;
    }
    ~BvhGpu() {
        delete left;
        delete right;
    }

    NGP_HOST_DEVICE vec3 center() const override { return _center; }
    NGP_HOST_DEVICE AABB bounding_box() const override { return _bounding_box; }

    __device__ bool hit(const Ray& orig_ray, float ray_tmin, float ray_tmax, HitRecord& rec) const override {
        Ray r {orig_ray.o - _center, orig_ray.d};
        if (!_bounding_box.hit(r, ray_tmin, ray_tmax))
            return false;

        bool hit_left = left->hit(r, ray_tmin, ray_tmax, rec);
        bool hit_right = right->hit(r, ray_tmin, hit_left ? rec.t : ray_tmax, rec);

        return hit_left || hit_right;
    }

    NGP_HOST_DEVICE void print(uint32_t indent = 0) const override {
        char* str_indent = new char[indent];
        for (uint32_t i = 0; i < indent; ++i) str_indent[i] = ' ';
        printf(">>%sBVH: {%f, %f, %f} -> {%f, %f, %f}\n", str_indent,
            _bounding_box.x[0], _bounding_box.y[0], _bounding_box.z[0],
            _bounding_box.x[1], _bounding_box.y[1], _bounding_box.z[1]);
        left->print(indent + 1);
        right->print(indent + 1);
        delete str_indent;
    }

    __host__ virtual Hittable** copy_to_gpu() const {
        // noop.
		return nullptr;
    }

private:
    Hittable* left;
    Hittable* right;
    vec3 _center;
    AABB _bounding_box;
};

}
}