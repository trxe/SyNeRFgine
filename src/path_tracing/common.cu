#include <neural-graphics-primitives/path-tracing/common.cuh>

namespace ngp {
namespace pt {

__global__ void rand_init_pixels(int resx, int resy, curandState *rand_state) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    if ((i >= resx) || (j >= resy)) return;
    int pixel_idx = j * resx + i;
    curand_init(PT_SEED, pixel_idx, 0, &rand_state[pixel_idx]);
}

}
}