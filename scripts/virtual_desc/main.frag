#version 140
in vec2 UVs;
out vec4 frag_color;
uniform ivec2 nerf_resolution;
uniform ivec2 syn_resolution;
uniform ivec2 full_resolution;
uniform sampler2D syn_rgba;
uniform sampler2D syn_depth;
uniform sampler2D nerf_rgba;
uniform sampler2D nerf_depth;

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

const float max_nd = 16384.0;

void main() {
    vec2 tex_coords = UVs;
    tex_coords.y = 1.0 - tex_coords.y;
    tex_coords = unwarp(tex_coords);
    vec4 view_coords = vec4(tex_coords, 1.0, 1.0);

    int syn_pixel_size = full_resolution.x / syn_resolution.x;
    int nerf_pixel_size = full_resolution.x / nerf_resolution.x;
    // vec2 nerf_pixel_size = vec2(float(nerf_resolution.x) / float(full_resolution.x), float(nerf_resolution.y) / float(full_resolution.y)) / full_resolution;
    // vec2 syn_pixel_size = vec2(float(syn_resolution.x) / float(full_resolution.x), float(syn_resolution.y) / float(full_resolution.y)) / full_resolution;

    // vec4 syn = texture(syn_rgba, tex_coords.xy);
    vec4 syn = textureProjOffset(syn_rgba, view_coords, ivec2(-1, +1) * syn_pixel_size)
             + textureProjOffset(syn_rgba, view_coords, ivec2(+1, -1) * syn_pixel_size)
             + textureProjOffset(syn_rgba, view_coords, ivec2(+1, +1) * syn_pixel_size)
             + textureProjOffset(syn_rgba, view_coords, ivec2(-1, -1) * syn_pixel_size);
    syn /= 4.0;
    float sd = texture(syn_depth, tex_coords.xy).r;
    vec4 nerf = texture(nerf_rgba, tex_coords.xy);
    // if (length(nerf.rgb) < 0.3) 
    //     nerf = textureProjOffset(nerf_rgba, view_coords, ivec2(-1, +1) * nerf_pixel_size)
    //             + textureProjOffset(nerf_rgba, view_coords, ivec2(+1, -1) * nerf_pixel_size)
    //             + textureProjOffset(nerf_rgba, view_coords, ivec2(+1, +1) * nerf_pixel_size)
    //             + textureProjOffset(nerf_rgba, view_coords, ivec2(-1, -1) * nerf_pixel_size);
    //     nerf /= 4.0;
    // }
    float nd = texture(nerf_depth, tex_coords.xy).r;

    if (sd < nd) {
        frag_color = vec4(syn.rgb, 1.0);
        gl_FragDepth = sd;
    } else if (nd < max_nd) {
        frag_color = vec4(nerf.rgb, 1.0);
        gl_FragDepth = nd;
    }
}