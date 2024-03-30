#include <synerfgine/common.cuh>
#include <neural-graphics-primitives/nerf_device.cuh>

namespace sng {

__global__ void debug_shade(uint32_t n_elements, vec4* __restrict__ rgba, vec3 color, float* __restrict__ depth, float depth_value) {
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_elements) return;
    rgba[idx] = color;
    depth[idx] = depth_value;
}

__global__ void print_shade(uint32_t n_elements, vec4* __restrict__ rgba, float* __restrict__ depth) {
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_elements) return;
    if (idx % 100 == 0) {
        vec4 color = rgba[idx];
        printf("%d: %f, %f, %f, %f | %f\n", idx, color.r, color.g, color.b, color.a, depth[idx]);
    }
}

__global__ void init_rand_state(uint32_t n_elements, curandState_t* rand_state) {
    uint64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_elements) return;
    curand_init(static_cast<uint64_t>(PT_SEED),  idx, (uint64_t)0, rand_state+idx);
}

__device__ float depth_test_world(const vec3& origin, const vec3& dir, const ObjectTransform* __restrict__ objects, const size_t& object_count, const int32_t& this_obj, int32_t& out_obj_id) {
    float depth = MAX_DEPTH();
    for (size_t c = 0; c < object_count; ++c) {
        if (c == this_obj) continue;
        ObjectTransform obj = objects[c];
        auto [tri_id, hit_d] = ngp::ray_intersect_nodes(origin, dir, obj.scale, obj.pos, obj.rot, obj.g_node, obj.g_tris);
        if (hit_d < depth && hit_d > MIN_DEPTH()) {
            out_obj_id = c;
            depth = hit_d;
        }
    }
    return depth;
}

__device__ float depth_test_world(const vec3& origin, const vec3& dir, const ObjectTransform* __restrict__ objects, const size_t& object_count, int32_t& out_obj_id, HitRecord& hit_info) {
    for (size_t c = 0; c < object_count; ++c) {
        ObjectTransform obj = objects[c];
        auto [tri_id, hit_d] = ngp::ray_intersect_nodes(origin, dir, obj.scale, obj.pos, obj.rot, obj.g_node, obj.g_tris);
        if (hit_d < hit_info.t && hit_d > MIN_DEPTH()) {
            out_obj_id = c;
            hit_info.t = hit_d;
            hit_info.material_idx = obj.mat_id;
            hit_info.normal = obj.rot * obj.g_tris[tri_id].normal();
            hit_info.perturb_matrix = obj.g_tris[tri_id].get_perturb_matrix();
            hit_info.object_idx = c;
        }
    }
    hit_info.pos = origin + hit_info.t * dir;
    hit_info.front_face = dot(dir, hit_info.normal) < 0.0;
    return hit_info.t;
}

__device__ float depth_test_nerf(const float& full_d, const uint32_t& n_steps, const float& cone_angle_constant, const vec3& next_pos, const vec3& L,
	const vec3& invL, const uint8_t* __restrict__ density_grid, const uint32_t& min_mip, const uint32_t& max_mip, const BoundingBox& render_aabb, const mat3& render_aabb_to_local
) {
	float nerf_shadow = 0.0f;
	for (uint32_t j = 0; j < n_steps; ++j) {
		nerf_shadow = if_unoccupied_advance_to_next_occupied_voxel(nerf_shadow, cone_angle_constant, {next_pos, L}, invL, density_grid, min_mip, max_mip, render_aabb, render_aabb_to_local);
		if (nerf_shadow >= full_d) {
			nerf_shadow = full_d;
			break;
		}
		float dt = calc_dt(nerf_shadow, cone_angle_constant);
		nerf_shadow += dt;
	}
	return nerf_shadow;
}

__device__ vec4 box_filter_vec4(uint32_t idx, ivec2 resolution, vec4* __restrict__ buffer, int kernel_size) {
    vec4 sum{};
    int nidx = idx;
    float z = 0.0;
    int x = idx % resolution.x, xmin = max(0, x-kernel_size), xmax = min(resolution.x-1, x+kernel_size);
    int y = idx / resolution.x, ymin = max(0, y-kernel_size), ymax = min(resolution.y-1, y+kernel_size);
    for (int i = xmin; i <= xmax; ++i) {
        for (int j = ymin; j <= ymax; ++j) {
            nidx = i * resolution.x + j;
            sum += buffer[nidx];
            z += 1.0f;
        }
    }
    return sum / z;
}

__global__ void transform_payload(
	uint32_t n_elements,
	const vec3* __restrict__ src_origin,
	const vec3* __restrict__ src_dir,
	const vec3* __restrict__ src_normal,
	vec3* __restrict__ dst_origin,
	vec3* __restrict__ dst_dir,
	vec3* __restrict__ dst_normal,
	mat3 rotation,
	vec3 translation,
	float scale_val,
	bool o2w
) {
	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= n_elements) return;
	if (!o2w) { // world space to object space
		mat3 scale = scale_val * mat3::identity();
		rotation = scale * rotation;
		dst_origin[idx] = rotation * (src_origin[idx] + translation);
		dst_dir[idx] = rotation * src_dir[idx];
		dst_normal[idx] = rotation * src_normal[idx];
	} else { // object space to world space
		dst_origin[idx] = rotation * (src_origin[idx] * scale_val) + translation;
		dst_dir[idx] = rotation * src_dir[idx] * scale_val;
		dst_normal[idx] = rotation * src_normal[idx] * scale_val;
	}
}

__global__ void transfer_color(
	uint32_t n_elements,
	const vec3* __restrict__ src_color,
	const float* __restrict__ src_depth,
	vec4* __restrict__ frame_buffer,
	float* __restrict__ depth_buffer
) {
	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= n_elements || src_color == nullptr || src_depth == nullptr) return;
	vec3 scol = src_color[idx];
	float sdep = src_depth[idx];
	float currdep = depth_buffer[idx];
	if (sdep < currdep) {
		frame_buffer[idx] = vec4(scol, 1.0);
		depth_buffer[idx] = sdep;
	}
}

__global__ void debug_uv_shade(uint32_t n_elements, vec4* __restrict__ rgba, float* __restrict__ depth, ivec2 resolution) {
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_elements) return;
    float x = (float)(idx % resolution.x) / (float) resolution.x;
    float y = (float)(idx / resolution.x) / (float) resolution.y;
    rgba[idx] = {x, y, 0.0, 1.0};
    depth[idx] = 1.0;
}

__device__ vec3 vec3_to_col(const vec3& v) {
    return v * 0.5f + vec3(0.5f);
}
}