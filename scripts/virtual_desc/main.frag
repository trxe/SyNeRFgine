#version 140
in vec2 UVs;
out vec4 frag_color;
uniform ivec2 nerf_resolution;
uniform ivec2 syn_resolution;
uniform ivec2 full_resolution;
uniform int filter_type;
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

#define MAX_ND 16384.0
#define MSIZE 16
#define SIGMA 16
#define BSIGMA 0.1

float kernel[MSIZE];

float normpdf(in float x, in float sigma) {
	return 0.39894*exp(-0.5*x*x/(sigma*sigma))/sigma;
}

float normpdf3(in vec3 v, in float sigma) {
	return 0.39894*exp(-0.5*dot(v,v)/(sigma*sigma))/sigma;
}

vec4 nerf_filter(int nerf_pixel_size, vec2 uv) {
    vec3 c = texture(nerf_rgba, uv).rgb;
    int kSize = (MSIZE - 1 / 2);
    vec4 view_coords = vec4(uv, 1.0, 1.0);
    for (int j = 0; j < kSize; ++j) {
        kernel[kSize + j] = kernel[kSize - j] = normpdf(float(j), SIGMA);
    }
    vec3 cc;
    float factor;
    float bZ = 1.0/normpdf(0.0, BSIGMA);
    vec3 final_colour;
    float Z = 0.0;
    for (int i=-kSize; i <= kSize; ++i)
    {
        for (int j=-kSize; j <= kSize; ++j)
        {
            // cc = texture(nerf_rgba, vec2(0.0, 1.0)-(uv+vec2(float(i),float(j))) / nerf_resolution.xy).rgb;
            cc = textureProjOffset(nerf_rgba, view_coords, ivec2(i, j) * nerf_pixel_size).rgb;
            factor = normpdf3(cc-c, BSIGMA)*bZ*kernel[kSize+j]*kernel[kSize+i];
            Z += factor;
            final_colour += factor*cc;

        }
    }
    return vec4(final_colour / Z, 1.0);
}

void main() {
    vec2 tex_coords = UVs;
    tex_coords.y = 1.0 - tex_coords.y;
    tex_coords = unwarp(tex_coords);
    vec4 view_coords = vec4(tex_coords, 1.0, 1.0);

    int syn_pixel_size = full_resolution.x / syn_resolution.x;
    int nerf_pixel_size = full_resolution.x / nerf_resolution.x;


    // antialiasing for syn
    vec4 syn = textureProjOffset(syn_rgba, view_coords, ivec2(-1, +1) * syn_pixel_size)
             + textureProjOffset(syn_rgba, view_coords, ivec2(+1, -1) * syn_pixel_size)
             + textureProjOffset(syn_rgba, view_coords, ivec2(+1, +1) * syn_pixel_size)
             + textureProjOffset(syn_rgba, view_coords, ivec2(-1, -1) * syn_pixel_size);
    syn /= 4.0;
    float sd = texture(syn_depth, tex_coords).r;
    vec4 nerf;
    switch (filter_type) {
    case 1:
        // nerf = nerf_filter(nerf_pixel_size, tex_coords);
        nerf = vec4(1.0, 0.0, 0.0, 1.0);
        break;
    default:
        nerf = texture(nerf_rgba, tex_coords);
        break;
    }
    float nd = texture(nerf_depth, tex_coords).r;

    if (sd < nd) {
        frag_color = vec4(syn.rgb, 1.0);
        gl_FragDepth = sd;
    } else if (nd < MAX_ND) {
        frag_color = vec4(nerf.rgb, 1.0);
        gl_FragDepth = nd;
    }
}