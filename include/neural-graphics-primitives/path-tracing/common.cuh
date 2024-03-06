#pragma once
#include <numeric>
#include <curand_kernel.h>
#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/common_device.cuh>
#define PT_SEED 1999

namespace ngp { 
namespace pt {

constexpr float PT_EPSILON = std::numeric_limits<float>::epsilon();
constexpr float PT_MIN_FLOAT = std::numeric_limits<float>::min();
constexpr float PT_MAX_FLOAT = std::numeric_limits<float>::max();

__global__ void rand_init_pixels(int resx, int resy, curandState *rand_state);

class Rand {
public:
    __host__ static inline vec3 random_unit_vector() {
        float x = (float)std::rand() / (float)RAND_MAX;
        float y = (float)std::rand() / (float)RAND_MAX;
        float z = (float)std::rand() / (float)RAND_MAX;
        vec3 rnd_vec = {x, y, z};
        return normalize(rnd_vec);
    }

    __device__ static inline vec3 random_unit_vector(curandState* random_seed) {
        vec3 tmp {
            curand_normal(random_seed),
            curand_normal(random_seed),
            curand_normal(random_seed)
        };
        return normalize(tmp);
    }

};

}
}