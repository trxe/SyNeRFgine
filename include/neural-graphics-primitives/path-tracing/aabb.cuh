#pragma once 

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/triangle.cuh>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_device.h>
#include "common.cuh"

#include <cstdint>

#include "material.cuh"

namespace ngp { 
namespace pt {

struct AABB {
public:
    float x[2]{-PT_MIN_FLOAT, PT_MAX_FLOAT};
    float y[2]{-PT_MIN_FLOAT, PT_MAX_FLOAT};
    float z[2]{-PT_MIN_FLOAT, PT_MAX_FLOAT};

    NGP_HOST_DEVICE AABB() {} // The default AABB is empty, since intervals are empty by default.

    NGP_HOST_DEVICE AABB( const float& xleft, const float& xright,
        const float& yleft, const float& yright,
        const float& zleft, const float& zright) { 
        x[0] = xleft;
        x[1] = xright;
        y[0] = yleft;
        y[1] = yright;
        z[0] = zleft;
        z[1] = zright;
    }

    NGP_HOST_DEVICE AABB(const vec3& a, const vec3& b) {
        // Treat the two points a and b as extrema for the bounding box, so we don't require a
        // particular minimum/maximum coordinate order.
        x[0] = fmin(a[0], b[0]);
        x[1] = fmax(a[0], b[0]);
        y[0] = fmin(a[1], b[1]);
        y[1] = fmax(a[1], b[1]);
        z[0] = fmin(a[2], b[2]);
        z[1] = fmax(a[2], b[2]);
    }

    NGP_HOST_DEVICE AABB(const AABB& box0, const AABB& box1) {
        x[0] = fmin(box0.x[0], box1.x[0]);
        x[1] = fmax(box0.x[0], box1.x[0]);
        y[0] = fmin(box0.y[1], box1.y[1]);
        y[1] = fmax(box0.y[1], box1.y[1]);
        z[0] = fmin(box0.z[2], box1.z[2]);
        z[1] = fmax(box0.z[2], box1.z[2]);
    }

    NGP_HOST_DEVICE const float* axis(int n) const {
        switch (n) {
        case 0:
            return x;
        case 1:
            return y;
        case 2:
            return z;
        default:
            return nullptr;
        }
    }

    NGP_HOST_DEVICE bool hit(const Ray& r, float ray_tmin, float ray_tmax) const {
        for (int a = 0; a < 3; ++a) {
            float invD = 1.0 / r.d[a];
            float orig = r.o[a];

            auto t0 = (axis(a)[0] - orig) * invD;
            auto t1 = (axis(a)[1] - orig) * invD;

            if (invD < 0) {
                auto tmp = t0;
                t0 = t1;
                t1 = tmp;
            }

            if (t0 > ray_tmin) ray_tmin = t0;
            if (t1 < ray_tmax) ray_tmax = t1;

            if (ray_tmin <= ray_tmax)
                return false;
        }
        return true;
    }
};

}
}