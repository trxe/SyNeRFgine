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
#define FXAA_EDGE_THRESHOLD 0.750
#define FXAA_EDGE_THRESHOLD_MIN 0.03125
float depth_edge_detection(vec4 view_coords, float center_depth, int kernel_size) {
    int syn_pixel_size = full_resolution.x / syn_resolution.x;
    float range_min = center_depth;
    float range_max = center_depth;
    for (int i = -kernel_size; i <= kernel_size; ++i) {
        for (int j = -kernel_size; j <= kernel_size; ++j) {
            float syn_d = textureProjOffset(syn_depth, view_coords, ivec2(i, j) * syn_pixel_size).r;
            range_min = min(syn_d, range_min);
            range_max = max(syn_d, range_max);
        }
    }
    float range = range_max - range_min;
    if (range < max(FXAA_EDGE_THRESHOLD, range_max * FXAA_EDGE_THRESHOLD_MIN)) {
        return 0.0;
    }
    return 1.0;
}

vec4 blur_kernel(vec4 view_coords, int kernel_size) {
    float factor = 0.0;
    vec4 final_color = vec4(0.0);
    for (int i = -kernel_size; i <= kernel_size; ++i) {
        for (int j = -kernel_size; j <= kernel_size; ++j) {
            float nerf_d = textureProjOffset(nerf_depth, view_coords, ivec2(i, j)).r;
            float syn_d = textureProjOffset(syn_depth, view_coords, ivec2(i, j)).r;
            vec4 val = syn_d < nerf_d ?
                textureProjOffset(syn_rgba, view_coords, ivec2(i, j)) :
                textureProjOffset(nerf_rgba, view_coords, ivec2(i, j));
            final_color += val;
            factor += 1.0;
        }
    }
    return final_color / float(factor);
}

void main() {
    int syn_pixel_size = full_resolution.x / syn_resolution.x;
    int nerf_pixel_size = full_resolution.x / nerf_resolution.x;
    vec2 tex_coords = UVs;
    tex_coords.y = 1.0 - tex_coords.y;
    tex_coords = unwarp(tex_coords);
    vec4 view_coords = vec4(tex_coords, 1.0, 1.0);
    vec4 syn = texture(syn_rgba, tex_coords);
    vec4 nerf = texture(nerf_rgba, tex_coords);
    float sd = texture(syn_depth, tex_coords).r;
    float nd = texture(nerf_depth, tex_coords).r;
    // float bal = (sd - nd >= 0.1) ? 0.0 : (sd - nd) + 0.5;

    // vec4 blur_color = blur_kernel(view_coords, 2);
    // vec3 syn_rgb = mix(
    //     syn.rgb, blur_color.rgb, max(bal, depth_edge_detection(view_coords, sd, 2))
    // );
    // frag_color = vec4(sd < nd ? syn.rgb : nerf.rgb, 1.0);
    frag_color = vec4(mix(syn.rgb, nerf.rgb, 0.0), 1.0);
    // frag_color = vec4(vec3(max(0.0, 1.0 - nd / 5.0)), 1.0);
    gl_FragDepth = sd;
}