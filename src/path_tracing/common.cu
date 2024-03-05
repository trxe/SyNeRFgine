#include <neural-graphics-primitives/path-tracing/common.cuh>

namespace ngp {
namespace pt {

__device__ vec3 vec3_to_col(const vec3& v) {
    return v * 0.5f + vec3(0.5f);
}

__device__ void print_vec(const vec3& v) {
    printf("%f, %f, %f\n", v.x, v.y, v.z);
}

__device__ void print_mat4x3(const mat4x3& camera) {
	printf("%f,\t%f,\t%f,\t%f\n%f,\t%f,\t%f,\t%f\n%f,\t%f,\t%f,\t%f\n",
		camera[0][0], camera[1][0], camera[2][0], camera[3][0],
		camera[0][1], camera[1][1], camera[2][1], camera[3][1],
		camera[0][2], camera[1][2], camera[2][2], camera[3][2]
	);
}

__global__ void rand_init_pixels(int resx, int resy, curandState *rand_state) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    if ((i >= resx) || (j >= resy)) return;
    int pixel_idx = j * resx + i;
    curand_init(PT_SEED, pixel_idx, 0, &rand_state[pixel_idx]);
}

}
}