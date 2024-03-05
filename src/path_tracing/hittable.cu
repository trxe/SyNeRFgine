#include <neural-graphics-primitives/path-tracing/hittable.cuh> 
#include <neural-graphics-primitives/path-tracing/hittable_list.cuh> 

namespace ngp {
namespace pt {

__global__ void pt_init_tri(Hittable** tri, vec3 p0, vec3 p1, vec3 p2, uint32_t mat_idx) {
    *tri = new Tri(p0, p1, p2,mat_idx);
}

__global__ void pt_init_bvh(Hittable** bvh, Hittable** left, Hittable** right, vec3 center, AABB bbox) {
    *bvh = new BvhGpu(left, right, center, bbox);
}

__global__ void pt_print_bvh(Hittable** bvh) {
    (*bvh)->print();
}

__global__ void pt_debug_hittables(uint32_t count, Hittable** bvh) {
    uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    printf("hittable %p\n", bvh[i]);
    if (bvh[i]) bvh[i]->print();
}

}
}