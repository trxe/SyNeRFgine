#include <neural-graphics-primitives/path-tracing/material.cuh>

namespace ngp {
namespace pt {

__global__ void init_material_lambertian(Material** __restrict__ d_mat, vec3 albedo) {
    *d_mat = new LambertianMaterial(albedo);
    printf("init : %p\n", *d_mat);
}

}
}