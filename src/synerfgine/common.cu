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

__device__ vec3 random_unit_vector(curandState_t* rand_state) {
	return normalize(vec3{
		fractf(curand_uniform(rand_state)),
		fractf(curand_uniform(rand_state)),
		fractf(curand_uniform(rand_state))
	});
}

__device__ float depth_test_world(const vec3& origin, const vec3& dir, const ObjectTransform* __restrict__ objects, const size_t& object_count, int32_t& out_obj_id) {
    float depth = MAX_DEPTH();
	const vec3 offset_origin = origin + dir * MIN_DEPTH();
    for (size_t c = 0; c < object_count; ++c) {
        ObjectTransform obj = objects[c];
        auto [tri_id, hit_d] = ngp::ray_intersect_nodes(offset_origin, dir, obj.scale, obj.pos, obj.rot, obj.g_node, obj.g_tris);
        if (hit_d < depth && hit_d > MIN_DEPTH()) {
            out_obj_id = c;
            depth = hit_d;
        }
    }
    return depth;
}

__device__ float depth_test_world(const vec3& origin, const vec3& dir, const ObjectTransform* __restrict__ objects, const size_t& object_count, int32_t& out_obj_id, HitRecord& hit_info) {
	const vec3 offset_origin = origin + dir * MIN_DEPTH();
    for (size_t c = 0; c < object_count; ++c) {
        ObjectTransform obj = objects[c];
        auto [tri_id, hit_d] = ngp::ray_intersect_nodes(offset_origin, dir, obj.scale, obj.pos, obj.rot, obj.g_node, obj.g_tris);
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

__device__ float depth_test_nerf(const float& full_d, const uint32_t& n_steps, const float& cone_angle_constant, const vec3& src, const vec3& L, const vec3& invL,
	const uint8_t* __restrict__ density_grid, const uint32_t& min_mip, const uint32_t& max_mip, const BoundingBox& render_aabb, const mat3& render_aabb_to_local
) {
	float nerf_shadow = 0.0f;
	for (uint32_t j = 0; j < n_steps; ++j) {
		nerf_shadow = if_unoccupied_advance_to_next_occupied_voxel(nerf_shadow, cone_angle_constant, {src, L}, invL, density_grid, min_mip, max_mip, render_aabb, render_aabb_to_local);
		if (nerf_shadow >= full_d) {
			nerf_shadow = full_d;
			break;
		}
		float dt = calc_dt(nerf_shadow, cone_angle_constant);
		nerf_shadow += dt;
	}
	return nerf_shadow;
}

__device__ float depth_test_nerf(const uint32_t& n_steps, const float& cone_angle_constant, const vec3& src, const vec3& dst,
	const uint8_t* __restrict__ density_grid, const uint32_t& min_mip, const uint32_t& max_mip, const BoundingBox& render_aabb, const mat3& render_aabb_to_local
) {
    float full_d = length(dst - src);
    vec3 L = normalize(dst - src);
    vec3 invL = 1.0f / L;
	float nerf_shadow = 0.0f;
	for (uint32_t j = 0; j < n_steps; ++j) {
		nerf_shadow = if_unoccupied_advance_to_next_occupied_voxel(nerf_shadow, cone_angle_constant, {src, L}, invL, density_grid, min_mip, max_mip, render_aabb, render_aabb_to_local);
		if (nerf_shadow >= full_d) {
			nerf_shadow = full_d;
			break;
		}
		float dt = calc_dt(nerf_shadow, cone_angle_constant);
		nerf_shadow += dt;
	}
	return nerf_shadow;
}

__device__ std::tuple<bool, float, float> depth_test_nerf_far(const uint32_t& n_steps, const float& cone_angle_constant, const vec3& src, const vec3& dst,
	const uint8_t* __restrict__ density_grid, const uint32_t& min_mip, const uint32_t& max_mip, const BoundingBox& render_aabb, const mat3& render_aabb_to_local
) {
	if (!density_grid) return {false, 0.0f, 0.0f};
    float full_d = length(dst - src);
    vec3 L = normalize(dst - src);
    vec3 invL = 1.0f / L;
	bool found_occupied = false;
	const float t = depth_test_nerf(n_steps, cone_angle_constant, src, dst, density_grid, min_mip, max_mip, render_aabb, render_aabb_to_local);
	if (!density_grid || t >= full_d) return {false, full_d, 0.0f};
	float nerf_shadow = t;
	bool empty_space_reached = false;
	for (uint32_t i = 0; i < n_steps; ++i) {
		while (true) {
			vec3 pos = src + nerf_shadow * L;
			if (nerf_shadow >= full_d) {
				// return {false, nerf_shadow, nerf_shadow - t}; // source pos intersected by something else
				break;
			}

			uint32_t mip = clamp(mip_from_pos(pos), min_mip, max_mip);
			if (!density_grid_occupied_at(pos, density_grid, mip)) {
				// return {true, nerf_shadow, nerf_shadow - t};
				empty_space_reached = true;
				break;
			}

			// Find largest empty voxel surrounding us, such that we can advance as far as possible in the next step.
			// Other places that do voxel stepping don't need this, because they don't rely on thread coherence as
			// much as this one here.
			while (mip < max_mip && density_grid_occupied_at(pos, density_grid, mip+1)) {
				++mip;
			}

			nerf_shadow = advance_to_next_voxel(nerf_shadow, cone_angle_constant, pos, L, invL, mip);
		}
		float dt = calc_dt(nerf_shadow, cone_angle_constant);
		nerf_shadow += dt;
	}
	return {empty_space_reached, min(full_d, nerf_shadow), nerf_shadow - t};
}

// Not used: results not favourable (similar  problem to going in inverse dierection)
__device__ std::pair<bool, float> depth_test_nerf_furthest(const uint32_t& n_steps, const float& cone_angle_constant, const vec3& src, const vec3& dst,
	const uint8_t* __restrict__ density_grid, const uint32_t& min_mip, const uint32_t& max_mip, const BoundingBox& render_aabb, const mat3& render_aabb_to_local,
	const float& threshold
) {
	uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (!density_grid) return {false, 0.0f};
    float full_d = length(dst - src);
    const vec3 L = normalize(dst - src);
	vec3 curr_src = src;
	float nerf_shadow = 0.0f;
	bool has_hit_something = false;
	while (true) {
		auto [has_hit_further, dist, thickness] = depth_test_nerf_far(n_steps, cone_angle_constant, curr_src, dst, density_grid, min_mip, max_mip, render_aabb, render_aabb_to_local);
		// if (i % 100000 == 0) printf("%d: %f | %f\n", i, thickness, threshold);
		if (!has_hit_further) return {has_hit_something, nerf_shadow};
		has_hit_something = has_hit_further;
		nerf_shadow += dist;
		curr_src += src + nerf_shadow * L;
		// if (i % 100000 == 0) printf("%d: d[%f] %f/%f\n", i, dist, nerf_shadow, full_d);
		if (thickness > threshold) return {has_hit_something, nerf_shadow};
	}
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

__device__ vec3 sng_tonemap(vec3 x, ETonemapCurve curve) {
	if (curve == ETonemapCurve::Identity) {
		return x;
	}

	x = max(x, vec3(0.0f));

	float k0, k1, k2, k3, k4, k5;
	if (curve == ETonemapCurve::ACES) {
		// Source:  ACES approximation : https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
		// Include pre - exposure cancelation in constants
		k0 = 0.6f * 0.6f * 2.51f;
		k1 = 0.6f * 0.03f;
		k2 = 0.0f;
		k3 = 0.6f * 0.6f * 2.43f;
		k4 = 0.6f * 0.59f;
		k5 = 0.14f;
	} else if (curve == ETonemapCurve::Hable) {
		// Source: https://64.github.io/tonemapping/
		const float A = 0.15f;
		const float B = 0.50f;
		const float C = 0.10f;
		const float D = 0.20f;
		const float E = 0.02f;
		const float F = 0.30f;
		k0 = A * F - A * E;
		k1 = C * B * F - B * E;
		k2 = 0.0f;
		k3 = A * F;
		k4 = B * F;
		k5 = D * F * F;

		const float W = 11.2f;
		const float nom = k0 * (W*W) + k1 * W + k2;
		const float denom = k3 * (W*W) + k4 * W + k5;
		const float white_scale = denom / nom;

		// Include white scale and exposure bias in rational polynomial coefficients
		k0 = 4.0f * k0 * white_scale;
		k1 = 2.0f * k1 * white_scale;
		k2 = k2 * white_scale;
		k3 = 4.0f * k3;
		k4 = 2.0f * k4;
	} else { //if (curve == ETonemapCurve::Reinhard)
		const vec3 luminance_coefficients = {0.2126f, 0.7152f, 0.0722f};
		float Y = dot(luminance_coefficients, x);

		return x * (1.f / (Y + 1.0f));
	}

	vec3 color_sq = x * x;
	vec3 nom = color_sq * k0 + k1 * x + k2;
	vec3 denom = k3 * color_sq + k4 * x + k5;

	vec3 tonemapped_color = nom / denom;

	return tonemapped_color;
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