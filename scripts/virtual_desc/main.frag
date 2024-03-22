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
uniform float nerf_shadow_blur_threshold;
uniform float nerf_blur_kernel_size;

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
#define SIGMA 8
#define BSIGMA 0.8

float kernel[MSIZE];

float normpdf(in float x, in float sigma) {
	return 0.39894*exp(-0.5*x*x/(sigma*sigma))/sigma;
}

float normpdf3(in vec3 v, in float sigma) {
	return 0.39894*exp(-0.5*dot(v,v)/(sigma*sigma))/sigma;
}

#define FXAA_SPAN_MAX	8.0
#define FXAA_REDUCE_MUL 1.0/8.0
#define FXAA_REDUCE_MIN 1.0/128.0

vec4 syn_draw(vec2 uv) {
    vec2 add = vec2(1.0) / syn_resolution.xy;
			
	vec3 rgbNW = texture(syn_rgba, uv+vec2(-add.x, -add.y)).rgb;
	vec3 rgbNE = texture(syn_rgba, uv+vec2( add.x, -add.y)).rgb;
	vec3 rgbSW = texture(syn_rgba, uv+vec2(-add.x,  add.y)).rgb;
	vec3 rgbSE = texture(syn_rgba, uv+vec2( add.x,  add.y)).rgb;
	vec3 rgbM  = texture(syn_rgba, uv).rgb;
	
	vec3 luma	 = vec3(0.299, 0.587, 0.114);
	float lumaNW = dot(rgbNW, luma);
	float lumaNE = dot(rgbNE, luma);
	float lumaSW = dot(rgbSW, luma);
	float lumaSE = dot(rgbSE, luma);
	float lumaM  = dot(rgbM,  luma);
	
	float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
	float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));
	
	vec2 dir;
	dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
	dir.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));
	
	
	float dirReduce = max(
		(lumaNW + lumaNE + lumaSW + lumaSE) * (0.25 * FXAA_REDUCE_MUL), FXAA_REDUCE_MIN);
	  
	float rcpDirMin = 1.0/(min(abs(dir.x), abs(dir.y)) + dirReduce);
	

	dir = min(vec2( FXAA_SPAN_MAX,  FXAA_SPAN_MAX),
		  max(vec2(-FXAA_SPAN_MAX, -FXAA_SPAN_MAX),
		  dir * rcpDirMin)) * add;

		
	vec3 rgbA = (1.0/2.0) * (texture(syn_rgba, uv + dir * (1.0/3.0 - 0.5)) +
							 texture(syn_rgba, uv + dir * (2.0/2.0 - 0.5))).rgb;
	
	vec3 rgbB = rgbA * (1.0/2.0) + (1.0/4.0) *
		(texture(syn_rgba, uv.xy + dir * (0.0/3.0 - 0.5)) +
		 texture(syn_rgba, uv.xy + dir * (3.0/3.0 - 0.5))).rgb;
	
	float lumaB = dot(rgbB, luma);
	if((lumaB < lumaMin) || (lumaB > lumaMax))
	{
        return vec4(rgbA, 1.0);
	}else
	{
        return vec4(rgbB, 1.0);
	}

}

vec3 bilateral_filter(sampler2D tex_rgba, int pixel_size, vec2 uv) {
    vec3 c = texture(tex_rgba, uv).rgb;
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
    for (int i=-kSize; i <= kSize; ++i) {
        for (int j=-kSize; j <= kSize; ++j) {
            cc = textureProjOffset(tex_rgba, view_coords, ivec2(i, j) * pixel_size).rgb;
            factor = normpdf3(cc-c, BSIGMA)*bZ*kernel[kSize+j]*kernel[kSize+i];
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
            // factor = normpdf3(cc-c, BSIGMA)*bZ*kernel[kSize+j]*kernel[kSize+i];
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
    return range;
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

#define INV_SQRT_OF_2PI 0.39894228040143267793994605993439  // 1.0/SQRT_OF_2PI
#define INV_PI 0.31830988618379067153776752674503
vec4 smartDeNoise(sampler2D tex, vec2 uv, float sigma, float kSigma, float threshold)
{
    float radius = round(kSigma*sigma);
    float radQ = radius * radius;

    float invSigmaQx2 = .5 / (sigma * sigma);      // 1.0 / (sigma^2 * 2.0)
    float invSigmaQx2PI = INV_PI * invSigmaQx2;    // 1/(2 * PI * sigma^2)

    float invThresholdSqx2 = .5 / (threshold * threshold);     // 1.0 / (sigma^2 * 2.0)
    float invThresholdSqrt2PI = INV_SQRT_OF_2PI / threshold;   // 1.0 / (sqrt(2*PI) * sigma^2)

    vec4 centrPx = texture(tex,uv); 

    float zBuff = 0.0;
    vec4 aBuff = vec4(0.0);
    vec2 size = vec2(textureSize(tex, 0));

    vec2 d;
    for (d.x=-radius; d.x <= radius; d.x++) {
        float pt = sqrt(radQ-d.x*d.x);       // pt = yRadius: have circular trend
        for (d.y=-pt; d.y <= pt; d.y++) {
            float blurFactor = exp( -dot(d , d) * invSigmaQx2 ) * invSigmaQx2PI;

            vec4 walkPx =  texture(tex,uv+d/size);
            vec4 dC = walkPx-centrPx;
            float deltaFactor = exp( -dot(dC, dC) * invThresholdSqx2) * invThresholdSqrt2PI * blurFactor;

            zBuff += deltaFactor;
            aBuff += deltaFactor*walkPx;
        }
    }
    return aBuff/zBuff;
}

vec3 main_filter(vec4 view_coords) {
    int syn_pixel_size = full_resolution.x / syn_resolution.x;
    int nerf_pixel_size = full_resolution.x / nerf_resolution.x;
    int full_pixel_size = 1;
    float nd[5];
    float sd[5];
    // float ndsum = 0.0, sdsum = 0.0;
    nd[0] = textureProjOffset(nerf_depth, view_coords, ivec2(0, -1)).r;
    nd[1] = textureProjOffset(nerf_depth, view_coords, ivec2(0,  1)).r;
    nd[2] = textureProjOffset(nerf_depth, view_coords, ivec2(0,  0)).r;
    nd[3] = textureProjOffset(nerf_depth, view_coords, ivec2( 1, 0)).r;
    nd[4] = textureProjOffset(nerf_depth, view_coords, ivec2(-1, 0)).r;

    sd[0] = textureProjOffset(syn_depth, view_coords, ivec2(0, -syn_pixel_size)).r;
    sd[1] = textureProjOffset(syn_depth, view_coords, ivec2(0,  syn_pixel_size)).r;
    sd[2] = textureProjOffset(syn_depth, view_coords, ivec2(0,  0)).r;
    sd[3] = textureProjOffset(syn_depth, view_coords, ivec2( syn_pixel_size, 0)).r;
    sd[4] = textureProjOffset(syn_depth, view_coords, ivec2(-syn_pixel_size, 0)).r;
    bool is_syn = sd[2] < nd[2];
    for (int i = 0; i < 5; ++i) {
        sd[i] = min(sd[i], nd[i]);
    }
    float drange_min = min(sd[0], min(min(sd[1], sd[2]), min(sd[3], sd[4])));
    float drange_max = max(sd[0], max(max(sd[1], sd[2]), max(sd[3], sd[4])));
    float drange = drange_max - drange_min;
    if (drange < max(FXAA_EDGE_THRESHOLD, drange_max * FXAA_EDGE_THRESHOLD)) {
        drange = 0.0;
    }
    nd[0] = luma(textureProjOffset(nerf_rgba, view_coords, ivec2(0, -nerf_pixel_size)));
    nd[1] = luma(textureProjOffset(nerf_rgba, view_coords, ivec2(0,  nerf_pixel_size)));
    nd[2] = luma(textureProjOffset(nerf_rgba, view_coords, ivec2(0,  0)));
    nd[3] = luma(textureProjOffset(nerf_rgba, view_coords, ivec2( nerf_pixel_size, 0)));
    nd[4] = luma(textureProjOffset(nerf_rgba, view_coords, ivec2(-nerf_pixel_size, 0)));
    float srange_min = min(nd[0], min(min(nd[1], nd[2]), min(nd[3], nd[4])));
    float srange_max = max(nd[0], max(max(nd[1], nd[2]), max(nd[3], nd[4])));
    float srange = srange_max - srange_min;
    if (srange < max(FXAA_EDGE_THRESHOLD, srange_max * FXAA_EDGE_THRESHOLD) * 12.0) {
        srange = 0.0;
    }

    vec3 nerf_blend = box_filter(nerf_rgba, nerf_pixel_size, view_coords.xy, 2).rgb;
    vec3 nerf_denoise = smartDeNoise(nerf_rgba, view_coords.xy, 2.0, nerf_pixel_size, 0.2).rgb;
    // vec3 blur_col = vec3(0.0, 0.0, 1.0);
    // vec3 orig_col = is_syn ? texture(syn_rgba, view_coords.xy).rgb :  texture(nerf_rgba, view_coords.xy).rgb;
    // float range = max(drange, srange);
    return nerf_denoise;
    if (srange > 0.0 && !is_syn) {
        // vec3 orig_col = texture(nerf_rgba, view_coords.xy).rgb;
        // return nerf_blend;
        return nerf_denoise;
        // return vec3(0.0, 0.0, 1.0);
    } else {
        vec3 orig_col = is_syn ? texture(syn_rgba, view_coords.xy).rgb :  texture(nerf_rgba, view_coords.xy).rgb;
        return orig_col;
    }
    // if (range == 0.0) return orig_col;
    // return vec3(1.0);
    // return mix(orig_col, blur_col, range) * 0.5  + vec3(0.5 * range);
    // return mix(orig_col, blur_col, range);
    // return vec3(range);
}

void main() {
    int syn_pixel_size = full_resolution.x / syn_resolution.x;
    int nerf_pixel_size = full_resolution.x / nerf_resolution.x;
    vec2 tex_coords = UVs;
    tex_coords.y = 1.0 - tex_coords.y;
    tex_coords = unwarp(tex_coords);
    vec4 view_coords = vec4(tex_coords, 1.0, 1.0);
    float sd = texture(syn_depth, tex_coords).r;
    float nd = texture(nerf_depth, tex_coords).r;

    // frag_color = vec4( sd < nd ? syn : nerf, 1.0 );

    // antialiasing for syn
    // vec4 syn;
    // syn = sd < nd ? msyn : texture(nerf_rgba, tex_coords);
    // float edge_range = max(depth_edge_detection(view_coords), shade_edge_detection(view_coords));
    // vec3 col = main_filter(view_coords);
    // frag_color = vec4(col, 1.0);
    // vec3 nerf = smartDeNoise(nerf_rgba, view_coords.xy, 2.0, nerf_pixel_size, 0.2).rgb;

    // float d = shade_edge_detection(syn_rgba, syn_pixel_size, view_coords, 2.0);
    // vec3 syn = mix( bilateral_filter(syn_rgba, syn_pixel_size, tex_coords),
    //                 box_filter(syn_rgba, syn_pixel_size, view_coords.xy, 3).rgb,
    //                 d );

    // vec3 syn = box_filter(syn_rgba, syn_pixel_size, view_coords.xy, 3).rgb;
    // vec3 syn = bilateral_filter(syn_rgba, syn_pixel_size, tex_coords);
    vec3 syn = texture(syn_rgba, tex_coords).rgb;
    vec3 nerf = texture(nerf_rgba, tex_coords).rgb;
    // vec3 syn = smartDeNoise(syn_rgba, view_coords.xy, 2.0, syn_pixel_size, 0.2).rgb;

    // BOX LOW PASS
    // vec4 nerf_box = box_filter(nerf_rgba, nerf_pixel_size * nerf_expand_mult, view_coords.xy, nerf_blur_kernel_size);
    /*
    vec4 nerf_box = box_filter(nerf_rgba, nerf_pixel_size * 3, view_coords.xy, 6);
    float lnb = luma(nerf_box);
    vec3 nerf;
    // if (luma(nerf_box) < nerf_shadow_blur_threshold) {
    if (luma(nerf_box) < 0.6) {
        nerf = mix(vec3(0.0), texture(nerf_rgba, tex_coords).rgb, lnb);
    } else {
        nerf = texture(nerf_rgba, tex_coords).rgb;
    }
    */
    frag_color = vec4( sd < nd ? syn : nerf, 1.0 );
    // frag_color = vec4(syn, 1.0);
    gl_FragDepth = sd;
}