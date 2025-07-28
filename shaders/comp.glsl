#version 450
//precision mediump float;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D rawimg;
layout(set = 0, binding = 1, rgba16) uniform writeonly image2D debayered;

layout (constant_id = 0) const int X_RED = 0;
layout (constant_id = 1) const int Y_RED = 0;
layout (constant_id = 2) const float BLACK_LEVEL = 0.0;
layout (constant_id = 3) const float STRETCH = 1.0;

#include "colorspaces.h"
#include "debayer.h"

layout(push_constant) uniform PushConstantData {
	mat3 cam_matrix;
	float exposure;
	float saturation_global;
	float saturation_shd, saturation_mid, saturation_hig;
	float white_re, black_re;
	float contrast;
} params;

float sq(const float x) {
	return x*x;
}

vec3 opacity_masks(const float x,
		   const float shadows_weight, const float highlights_weight,
                   const float midtones_weight, const float mask_grey_fulcrum)
{
	const float x_offset = (x - mask_grey_fulcrum);
	const float x_offset_norm = x_offset / mask_grey_fulcrum;
	const float alpha = 1.f / (1.f + exp(x_offset_norm * shadows_weight));    // opacity of shadows
	const float beta = 1.f / (1.f + exp(-x_offset_norm * highlights_weight)); // opacity of highlights
	const float gamma = exp(-sq(x_offset) * midtones_weight / 4.f) * sq(1.0 - alpha) * sq(1.0 - beta) * 8.f; // opacity of midtones
	return vec3(alpha, gamma, beta);
}

float log_tonemapping(const float x,
	const float grey, const float black,
	const float dynamic_range) {
	// All we care about, ultimately is to output in [0; 1] normalized space for the spline
	return clamp((log2(x / grey) - black) / dynamic_range, 0.f, 1.f);
}

vec3 log_tonemapping(const vec3 x,
	const float grey, const float black,
	const float dynamic_range) {
	// All we care about, ultimately is to output in [0; 1] normalized space for the spline
	return clamp((log2(x / grey) - black) / dynamic_range, 0.f, 1.f);
}

float scurve(float norm, float c, vec3 tc, vec3 sc, float sl, float tl) {
	float td = pow(0.18, 1.0/2.2), sd = td + (sl - tl)*c;
        float scw = -(4.*sc.x + 3.*sc.y + 2.*sc.z);
        float scu = 1. - sc.x - sc.y - sc.z - scw;
	vec3 norms = norm * vec3(tc.x, 0, sc.x) + vec3(tc.y, 0, sc.y);
	norms = norm * norms + vec3(tc.z, 0, sc.z);
	norms = norm * norms + vec3(0, c, scw);
	norms = norm * norms + vec3(0, td - c * tl, scu);
	return dot(norms, vec3(norm < tl, tl <= norm && norm <= sl, sl < norm));
}

// 0-1 sRGB  from  0-1 linear
vec3 srgb_from_linear(vec3 linear) {
	bvec3 cutoff = lessThan(linear, vec3(0.0031308));
	vec3 lower = linear * vec3(12.92);
	vec3 higher = vec3(1.055) * pow(linear, vec3(1./2.4)) - vec3(0.055);
	return mix(higher, lower, vec3(cutoff));
}

void main() {
	uvec2 pos = gl_GlobalInvocationID.xy;

	const vec3 c_cam = (debayer_mhc(pos) - vec3(BLACK_LEVEL)) * STRETCH;
	vec3 col_xyz = params.cam_matrix * c_cam;
	col_xyz *= pow(2.0, params.exposure);

	vec3 jch = xyY_to_dt_UCS_JCH(XYZ_to_xyY(col_xyz), 1.0);
	if (isinf(params.saturation_global)) {
 		jch.y = 0.0;
 	} else {
		const vec3 opacity = opacity_masks(jch.x, 1.0, 1.0, 1.0, 0.1845);
		vec3 hcb = dt_UCS_JCH_to_HCB(jch);
    		const vec2 sincos = normalize(hcb.yz);
		const vec3 saturation_local = vec3(params.saturation_shd, params.saturation_mid, params.saturation_hig);
		const float a = params.saturation_global + dot(saturation_local, opacity);
		const vec2 pw = vec2(a * hcb.y, sincos.x * hcb.y + sincos.y * hcb.z);
		const mat2 inv = mat2(sincos.y, -sincos.x, sincos);
		jch = dt_UCS_HCB_to_JCH(vec3(hcb.x, inv * pw));
 	}
	const vec3 col0 = xyz_to_rec2020 * xyY_to_XYZ(dt_UCS_JCH_to_xyY(jch, 1.0));
	//const vec3 col0 = xyz_to_rec2020 * col_xyz;

	//float norm = max(max(col0.r, col0.g), col0.b);
	//norm = clamp(norm, 0.18*pow(2., -6), 0.18*pow(2., 5.2));
	//vec3 ratios = col0 / norm;

	//norm = log_tonemapping(norm, 0.18, params.black_re, params.white_re - params.black_re);
	vec3 col1 = log_tonemapping(col0, 0.18, params.black_re, params.white_re - params.black_re);

	float c = params.contrast;
	float sl = 6.0/11.2, tl = sl;
	float sd = pow(0.18, 1.0/2.2), td = sd;

	float tl2 = tl*tl, tl3 = tl2*tl, tl4 = tl3*tl;
	mat3 t_sys = mat3(
    		tl4, 4.*tl3, 12.*tl2,
    		tl3, 3.*tl2, 6.*tl,
    		tl2, 2.*tl, 2.
    	);
	vec3 tc = inverse(t_sys) * vec3(td, c, 0);

	float sl2 = sl*sl, sl3 = sl2*sl, sl4 = sl3*sl;
	mat3 s_sys = mat3(
    		sl4 - 4*sl + 3., 4.*(sl3 - 1.), 12.*sl2,
    		sl3 - 3*sl + 2., 3.*(sl2 - 1.), 6*sl,
    		sl2 - 2*sl + 1., 2.*(sl - 1.), 2.
    	);
	vec3 sc = inverse(s_sys) * vec3(sd - 1.0, c, 0);

	/*vec3 norms = norm * vec3(tc.x, 0, sc.x) + vec3(tc.y, 0, sc.y);
	norms = norm * norms + vec3(tc.z, 0, sc.z);
	norms = norm * norms + vec3(0, c, sc.w);
	norms = norm * norms + vec3(0, td - c * tl, 1 - sc.x - sc.y - sc.z - sc.w);
	norm = dot(norms, vec3(norm < tl, tl <= norm && norm <= sl, sl < norm));
	
	norm = pow(norm, 2.2);
	vec3 col1 = norm * ratios;*/

	col1.x = scurve(col1.x, c, tc, sc, sl, tl);
	col1.y = scurve(col1.y, c, tc, sc, sl, tl);
	col1.z = scurve(col1.z, c, tc, sc, sl, tl);
	col1 = pow(col1, vec3(2.2));

	vec3 ych0 = rec2020_to_Ych(col0);
	vec3 ych1 = rec2020_to_Ych(col1);
	ych1 = vec3(ych1.x, min(ych0.y, ych1.y), ych0.z);
	col1 = Ych_to_rec2020(ych1);
	//col1 = vec3(ych1.x);

	col1 = (inverse(rec709_to_rec2020) * col1);

	col1 = srgb_from_linear(col1);
	imageStore(debayered, ivec2(pos), vec4(col1, 1.0));
}
