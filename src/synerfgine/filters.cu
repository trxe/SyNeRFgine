#include "synerfgine/filters.cuh"
#define PI 3.141592f

__device__ float median_kernel(float* __restrict__ values, ivec2 resolution, int x, int y, int inner_radius, int outer_radius) {
	float gauss_coeff = 0.0f;
    int count = 0;
	for (int di = inner_radius; di < outer_radius; ++di) {
		for (int dj = inner_radius; dj < outer_radius; ++dj) {
            int i = x + di;
            int j = y + dj;
			if (i >= 0 && i < resolution.x && j >= 0 && j < resolution.y) {
				int nidx = j * resolution.x + i;
				gauss_coeff += values[nidx];
                ++count;
			}
            i = x - di;
            j = y + dj;
			if (i >= 0 && i < resolution.x && j >= 0 && j < resolution.y) {
				int nidx = j * resolution.x + i;
				gauss_coeff += values[nidx];
                ++count;
			}
            i = x + di;
            j = y - dj;
			if (i >= 0 && i < resolution.x && j >= 0 && j < resolution.y) {
				int nidx = j * resolution.x + i;
				gauss_coeff += values[nidx];
                ++count;
			}
            i = x - di;
            j = y - dj;
			if (i >= 0 && i < resolution.x && j >= 0 && j < resolution.y) {
				int nidx = j * resolution.x + i;
				gauss_coeff += values[nidx];
                ++count;
			}
		}
	}
	return gauss_coeff / (float)(count);
}

__device__ float gaussian_kernel(float* __restrict__ values, ivec2 resolution, int x, int y, int kernel_size, float SD) {
	float gauss_coeff = 0.0f;
	for (int i = x - kernel_size; i <= x + kernel_size; ++i) {
		for (int j = y - kernel_size; j <= y + kernel_size; ++j) {
			if (i >= 0 && i < resolution.x && j >= 0 && j < resolution.y) {
				int nidx = j * resolution.x + i;
				int dx = i - x, dy = j - y;
				float g_coeff = exp(-(dx * dx + dy * dy)/ (2.0f * SD * SD)) / (2 * PI * SD * SD);
				gauss_coeff += g_coeff * values[nidx];
			}
		}
	}
	return gauss_coeff;
}

__device__ float pull_push_kernel(float* __restrict__ values, ivec2 resolution, int x, int y) {
    int idx = y * resolution.x + x;
    float orig = values[idx];
    // downsample
    ivec2 res0 = resolution;
    ivec2 uv = {x * 2 / res0.x, y * 2 / res0.y};
    ivec2 ofs = {res0.x / 2, res0.y / 2};
    int pos[4] = {
        {min(x + 1, res0.x - 1) + res0.x * max(y - 1, 0)},
        {max(x - 1, 0) + res0.x * max(y - 1, 0)},
        {min(x + 1, res0.x - 1) + res0.x * min(y + 1, res0.y - 1)},
        {max(x - 1, 0) + res0.x * min(y + 1, res0.y - 1)}
    };

    // PULL DOWNSAMPLE 1/2. no weights in initial pass
    float c0 = values[pos[0]];
    float c1 = values[pos[1]];
    float c2 = values[pos[2]];
    float c3 = values[pos[3]];
    vec4 w = vec4( 
        c0 > 0.0 ? 1.0 : 0.0,
        c1 > 0.0 ? 1.0 : 0.0,
        c2 > 0.0 ? 1.0 : 0.0,
        c3 > 0.0 ? 1.0 : 0.0         
    );
    float sumw1 = w[0] + w[1] + w[2] + w[3];
    float pullds1 = (c0 + c1 + c2 + c3) / sumw1;
    values[idx] = pullds1;
    __syncthreads();

    // PULL DOWNSAMPLE 2/2. weights in 2nd pass
    float pullds2 = 0.0;
    c0 = values[pos[0]];
    c1 = values[pos[1]];
    c2 = values[pos[2]];
    c3 = values[pos[3]];
    vec4 w2 = vec4( 
        c0 > 0.0 ? 1.0 : 0.0,
        c1 > 0.0 ? 1.0 : 0.0,
        c2 > 0.0 ? 1.0 : 0.0,
        c3 > 0.0 ? 1.0 : 0.0         
    );
    float sumw2 = w2[0] + w2[1] + w2[2] + w2[3];
    if ( sumw2 > 0.0 ) {
        pullds2 += c0 * w2[0] + c1 * w2[1] + c2 * w2[2] + c3 * w2[3];
        pullds2 /= sumw2;
    }
    values[idx] = pullds2;
    __syncthreads();

    // PUSH UPSAMPLE 
    const vec4 upsample_w = vec4(9.0/16.0, 3.0/16.0, 3.0/16.0, 1.0/16.0);
    pos[0] = {x + res0.x * y};
    pos[1] = {min(x + 1, res0.x) + res0.x * y};
    pos[2] = {x + res0.x * min(y + 1, res0.y - 1)};
    pos[3] = {max(x + 1, res0.x) + res0.x * min(y + 1, res0.y - 1)};
    vec4 g1 = vec4 ( values[pos[0]], values[pos[1]], values[pos[2]], values[pos[3]]);
    float weight_sum = dot(upsample_w, w);
    float pushus = dot(upsample_w * w, g1) / weight_sum;
    float mixval = min(1.0, sumw1);
    values[idx] = mixval * pushus + (1.0 - mixval) * pullds1;
    __syncthreads();
    float val = values[idx];
    values[idx] = orig;

    return val;
}

__device__ vec4 pull_push_kernel_rgba(vec4* __restrict__ values, ivec2 resolution, int x, int y, int rounds) {
    int idx = y * resolution.x + x;
    vec4 orig = values[idx];

    ivec2 res0 = resolution;
    int pos[4] = {
        {min(x + 1, res0.x - 1) + res0.x * max(y - 1, 0)},
        {max(x - 1, 0) + res0.x * max(y - 1, 0)},
        {min(x + 1, res0.x - 1) + res0.x * min(y + 1, res0.y - 1)},
        {max(x - 1, 0) + res0.x * min(y + 1, res0.y - 1)}
    };

    // PULL DOWNSAMPLE 1/2. no weights in initial pass
    vec4 c0 = values[pos[0]];
    vec4 c1 = values[pos[1]];
    vec4 c2 = values[pos[2]];
    vec4 c3 = values[pos[3]];
    vec4 c = vec4(0.0);
    c.a = c0.a + c1.a + c2.a + c3.a;
    if (c.a > 0.0) {
        c.rgb() += c0.rgb() * c0.a;
        c.rgb() += c1.rgb() * c1.a;
        c.rgb() += c2.rgb() * c2.a;
        c.rgb() += c3.rgb() * c3.a;
        c.rgb() / c.a;
    }
    values[idx] = c;
    __syncthreads();

    // PULL DOWNSAMPLE 2/2. weights in 2nd pass
    vec4 pullds2 = 0.0;
    c0 = values[pos[0]];
    c1 = values[pos[1]];
    c2 = values[pos[2]];
    c3 = values[pos[3]];
    vec4 w2 = vec4( 
        min(1.0, c0.a),
        min(1.0, c1.a),
        min(1.0, c2.a),
        min(1.0, c3.a)
    );
    float sumw2 = w2[0] + w2[1] + w2[2] + w2[3];
    if ( sumw2 > 0.0 ) {
        pullds2.rgb() += c0.rgb() * w2[0] + c1.rgb() * w2[1] + c2.rgb() * w2[2] + c3.rgb() * w2[3];
        pullds2.a = sumw2;
        pullds2.rgb() /= sumw2;
    }
    values[idx] = pullds2;
    __syncthreads();

    // PUSH UPSAMPLE 
    const vec4 upsample_w = vec4(9.0/16.0, 3.0/16.0, 3.0/16.0, 1.0/16.0);
    pos[0] = {x + res0.x * y};
    pos[1] = {min(x + 1, res0.x) + res0.x * y};
    pos[2] = {x + res0.x * min(y + 1, res0.y - 1)};
    pos[3] = {max(x + 1, res0.x) + res0.x * min(y + 1, res0.y - 1)};
    c0 = values[pos[0]];
    c1 = values[pos[1]];
    c2 = values[pos[2]];
    c3 = values[pos[3]];
    c0.a *= upsample_w[0];
    c1.a *= upsample_w[1];
    c2.a *= upsample_w[2];
    c3.a *= upsample_w[3];
    vec4 last_val = vec4(
        c0.rgb() * c0.a +
        c1.rgb() * c1.a +
        c2.rgb() * c2.a +
        c3.rgb() * c3.a, c0.a + c1.a + c2.a + c3.a
    );
    if (last_val.a > 0.0) {
        last_val.rgb() /= last_val.a;
        float k = min(1.0, last_val.a);
        last_val = last_val * (1.0f - k) + c * k;
    }
    __syncthreads();
    return last_val;
}