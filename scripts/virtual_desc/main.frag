#version 330 core

in vec2 UVs;
out vec4 frag_color;
uniform sampler2D syn_rgba;
uniform sampler2D syn_depth;
uniform sampler2D nerf_rgba;
uniform sampler2D nerf_depth;
uniform ivec2 nerf_resolution;
uniform ivec2 syn_resolution;
uniform ivec2 full_resolution;
uniform int filter_type;

uniform int nerf_expand_mult;
uniform int nerf_blur_kernel_size;
uniform float nerf_shadow_blur_threshold;

uniform int syn_blur_kernel_size;
uniform float syn_sigma;
uniform float syn_bsigma;

struct FoveationWarp {
    float al, bl, cl;
    float am, bm;
    float ar, br, cr;
    float switch_left, switch_right;
    float inv_switch_left, inv_switch_right;
};

uniform FoveationWarp warp_x;
uniform FoveationWarp warp_y;

float unwarp(in FoveationWarp warp, float y) {
    y = clamp(y, 0.0, 1.0);
    if (y < warp.inv_switch_left) {
        return (sqrt(-4.0 * warp.al * warp.cl + 4.0 * warp.al * y + warp.bl * warp.bl) - warp.bl) / (2.0 * warp.al);
    } else if (y > warp.inv_switch_right) {
        return (sqrt(-4.0 * warp.ar * warp.cr + 4.0 * warp.ar * y + warp.br * warp.br) - warp.br) / (2.0 * warp.ar);
    } else {
        return (y - warp.bm) / warp.am;
    }
}

vec2 unwarp(in vec2 pos) {
    return vec2(unwarp(warp_x, pos.x), unwarp(warp_y, pos.y));
}

#define MAX_ND 16384.0

float normpdf(in float x, in float sigma) {
	return 0.39894*exp(-0.5*x*x/(sigma*sigma))/sigma;
}

float normpdf3(in vec3 v, in float sigma) {
	return 0.39894*exp(-0.5*dot(v,v)/(sigma*sigma))/sigma;
}

#define FXAA_SPAN_MAX	8.0
#define FXAA_REDUCE_MUL 1.0/8.0
#define FXAA_REDUCE_MIN 1.0/128.0

vec3 bilateral_filter(sampler2D tex_rgba, int pixel_size, vec2 uv, int kernel_size) {
    vec3 c = texture(tex_rgba, uv).rgb;
    vec4 view_coords = vec4(uv, 1.0, 1.0);
    vec3 cc;
    float factor;
    float bZ = 1.0/normpdf(0.0, syn_bsigma);
    vec3 final_colour;
    float Z = 0.0;
    for (int i= -kernel_size; i <= kernel_size; ++i) {
        for (int j= -kernel_size; j <= kernel_size; ++j) {
            float norm_coeff_i = normpdf(float(abs(i)), syn_sigma);
            float norm_coeff_j = normpdf(float(abs(j)), syn_sigma);
            cc = textureProjOffset(tex_rgba, view_coords, ivec2(i, j) * pixel_size).rgb;
            factor = normpdf3(cc-c, syn_bsigma)*bZ*norm_coeff_i*norm_coeff_j;
            Z += factor;
            final_colour += factor*cc;
        }
    }
    final_colour /= Z;
    return final_colour;
}

vec4 box_filter(sampler2D tex_rgba, int pixel_size, vec2 uv, int kSize) { 
    vec4 view_coords = vec4(uv, 1.0, 1.0);
    vec3 final_colour = vec3(0.0);
    vec3 cc = vec3(0.0);
    float Z = 0.0;
    float factor = 0.0;
    for (int i=-kSize; i <= kSize; ++i) {
        for (int j=-kSize; j <= kSize; ++j) {
            cc = textureProjOffset(tex_rgba, view_coords, ivec2(i, j) * pixel_size).rgb;
            factor = 1.0;
            Z += factor;
            final_colour += factor*cc;
        }
    }
    return vec4(final_colour / Z, 1.0);
}

#define FXAA_EDGE_THRESHOLD 0.0625
#define FXAA_EDGE_THRESHOLD_MIN 0.03125
float depth_edge_detection(vec4 view_coords) {
    int syn_pixel_size = full_resolution.x / syn_resolution.x;
    int nerf_pixel_size = full_resolution.x / nerf_resolution.x;
    int full_pixel_size = 1;
    float nd[5];
    float sd[5];
    // float ndsum = 0.0, sdsum = 0.0;
    nd[0] = textureProjOffset(nerf_depth, view_coords, ivec2(0, -full_pixel_size)).r;
    nd[1] = textureProjOffset(nerf_depth, view_coords, ivec2(0,  full_pixel_size)).r;
    nd[2] = textureProjOffset(nerf_depth, view_coords, ivec2(0,  0)).r;
    nd[3] = textureProjOffset(nerf_depth, view_coords, ivec2( full_pixel_size, 0)).r;
    nd[4] = textureProjOffset(nerf_depth, view_coords, ivec2(-full_pixel_size, 0)).r;

    sd[0] = textureProjOffset(syn_depth, view_coords, ivec2(0, -syn_pixel_size)).r;
    sd[1] = textureProjOffset(syn_depth, view_coords, ivec2(0,  syn_pixel_size)).r;
    sd[2] = textureProjOffset(syn_depth, view_coords, ivec2(0,  0)).r;
    sd[3] = textureProjOffset(syn_depth, view_coords, ivec2( syn_pixel_size, 0)).r;
    sd[4] = textureProjOffset(syn_depth, view_coords, ivec2(-syn_pixel_size, 0)).r;
    for (int i = 0; i < 5; ++i) {
        // sdsum += sd[i];
        // ndsum += nd[i];
        sd[i] = min(sd[i], nd[i]);
    }
    float range_min = min(sd[0], min(min(sd[1], sd[2]), min(sd[3], sd[4])));
    float range_max = max(sd[0], max(max(sd[1], sd[2]), max(sd[3], sd[4])));
    float range = range_max - range_min;
    if (range < max(FXAA_EDGE_THRESHOLD, range_max * FXAA_EDGE_THRESHOLD)) {
        return 0.0;
    }
    return 1.0;
}

float luma(vec4 rgba) { return rgba.y * (0.587/0.299) + rgba.x; }

float shade_edge_detection(sampler2D rgba, int pixel_size, vec4 view_coords, float threshold_scale) {
    float nd[5];
    nd[0] = luma(textureProjOffset(rgba, view_coords, ivec2(0, -pixel_size)));
    nd[1] = luma(textureProjOffset(rgba, view_coords, ivec2(0,  pixel_size)));
    nd[2] = luma(textureProjOffset(rgba, view_coords, ivec2(0,  0)));
    nd[3] = luma(textureProjOffset(rgba, view_coords, ivec2( pixel_size, 0)));
    nd[4] = luma(textureProjOffset(rgba, view_coords, ivec2(-pixel_size, 0)));

    float range_min = min(nd[0], min(min(nd[1], nd[2]), min(nd[3], nd[4])));
    float range_max = max(nd[0], max(max(nd[1], nd[2]), max(nd[3], nd[4])));
    float range = range_max - range_min;
    if (range < max(FXAA_EDGE_THRESHOLD, range_max * FXAA_EDGE_THRESHOLD) * threshold_scale) {
    // if (range < max(FXAA_EDGE_THRESHOLD, range_max * FXAA_EDGE_THRESHOLD) * 13.0) {
        return 0.0;
    }
    return 1.0;
}

void main() {
    int syn_pixel_size = full_resolution.x / syn_resolution.x;
    int nerf_pixel_size = full_resolution.x / nerf_resolution.x;
    vec2 tex_coords = UVs;
    tex_coords.y = 1.0 - tex_coords.y;
    tex_coords = unwarp(tex_coords);
    vec4 view_coords = vec4(tex_coords, 1.0, 1.0);
    // vec3 syn = texture(syn_rgba, tex_coords).rgb;
    // vec3 nerf = texture(nerf_rgba, tex_coords).rgb;
    float sd = texture(syn_depth, tex_coords).r;
    float nd = texture(nerf_depth, tex_coords).r;

    float d = shade_edge_detection(syn_rgba, syn_pixel_size, view_coords, 2.0);
    vec3 syn = mix( bilateral_filter(syn_rgba, syn_pixel_size, tex_coords, syn_blur_kernel_size),
                    box_filter(syn_rgba, syn_pixel_size, view_coords.xy, syn_blur_kernel_size).rgb,
                    d );

    // vec3 syn = box_filter(syn_rgba, syn_pixel_size, view_coords.xy, syn_blur_kernel_size).rgb;
    // vec3 syn = bilateral_filter(syn_rgba, syn_pixel_size, tex_coords, syn_blur_kernel_size);
    // vec3 syn = smartDeNoise(syn_rgba, view_coords.xy, 2.0, syn_pixel_size, 0.2).rgb;

    // BOX LOW PASS
    vec4 nerf_box = box_filter(nerf_rgba, nerf_pixel_size * nerf_expand_mult, view_coords.xy, nerf_blur_kernel_size);
    // vec4 nerf_box = box_filter(nerf_rgba, nerf_pixel_size * 3, view_coords.xy, 6);
    float lnb = luma(nerf_box);
    vec3 nerf;
    if (luma(nerf_box) < nerf_shadow_blur_threshold) {
    // if (luma(nerf_box) < 0.6) {
        nerf = mix(vec3(0.0), texture(nerf_rgba, tex_coords).rgb, lnb);
    } else {
        nerf = texture(nerf_rgba, tex_coords).rgb;
    }
    frag_color = vec4( sd < nd ? syn : nerf, 1.0 );
    // frag_color = vec4(syn, 1.0);
    gl_FragDepth = sd;
}