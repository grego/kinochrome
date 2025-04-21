const mat3 rec2020_to_xyz = mat3(
	6.36958048e-01, 2.62700212e-01, 4.20575872e-11,
	1.44616904e-01, 6.77998072e-01, 2.80726931e-02,
	1.68880975e-01, 5.93017165e-02, 1.06098506e+00
);

const mat3 xyz_to_rec2020 = mat3(
	1.7166511880, -0.6666843518, 0.0176398574, 
	-0.3556707838, 1.6164812366, -0.0427706133, 
	-0.2533662814, 0.0157685458, 0.9421031212
);


const mat3 rec709_to_rec2020 = mat3(
	0.62750375, 0.06910828, 0.01639406,
	0.32927542, 0.91951916, 0.08801125,
	0.04330266, 0.0113596 , 0.89538035
);

const mat3 XYZ_D65_to_LMS_2006_D65 = mat3(
	0.257085, -0.394427, 0.064856,
	0.859943, 1.175800, -0.076250,
	-0.031061, 0.106423, 0.559067 
);

const mat3 LMS_2006_D65_to_XYZ_D65 = mat3(
	1.80794659, 0.61783960, -0.12546960,
	-1.29971660, 0.39595453, 0.20478038,
	0.34785879, -0.04104687, 1.74274183
);

const mat3 filmlightRGB_D65_to_LMS_D65 = mat3(
	0.95, 0.05, 0.00,
	0.38, 0.62, 0.00,
	0.00, 0.03, 0.97
);

const mat3 LMS_D65_to_filmlightRGB_D65 = mat3(
	1.0877193, -0.0877193, 0.,
	-0.66666667, 1.66666667, 0.,
	0.02061856, -0.05154639,  1.03092784 
);

vec3 LMS_to_Yrg(const vec3 LMS) {
	// compute luminance
	const float Y = 0.68990272 * LMS.x + 0.34832189 * LMS.y;

	// normalize LMS
	const float a = LMS.x + LMS.y + LMS.z;
	const vec3 lms = (a == 0.) ? vec3(0.) : LMS / a;

	// convert to Filmlight rgb (normalized)
	const vec3 rgb = LMS_D65_to_filmlightRGB_D65 * lms;

	return vec3(Y, rgb.x, rgb.y);
}

vec3 Yrg_to_LMS(const vec3 Yrg) {
	const float Y = Yrg.x;
	// reform rgb (normalized) from chroma
	const vec3 rgb = vec3(Yrg.yz, 1. - Yrg.y - Yrg.z);
	// convert to lms (normalized)
	const vec3 lms = filmlightRGB_D65_to_LMS_D65 * rgb;
	// denormalize to LMS
	const float denom = (0.68990272 * lms.x + 0.34832189 * lms.y);
	const float a = (denom == 0.) ? 0. : Y / denom;
	return lms * a;
}

/*
* Re-express Filmlight Yrg in polar coordinates Ych
*/
vec3 Yrg_to_Ych(const vec3 Yrg) {
	const float Y = Yrg.x;
	// Subtract white point. These are the r, g coordinates of
	// sRGB (D50 adapted) (1, 1, 1) taken through
	// XYZ D50 -> CAT16 D50->D65 adaptation -> LMS 2006
	// -> grading RGB conversion.
	const float r = Yrg.y - 0.21902143;
	const float g = Yrg.z - 0.54371398;
	const float c = sqrt(g*g + r*r);
	const float h = atan(g, r);
	return vec3(Y, c, h);
}

vec3 Ych_to_Yrg(const vec3 Ych) {
	const float Y = Ych.x;
	const float c = Ych.y;
	const float h = Ych.z;
	const float r = c * cos(h) + 0.21902143;
	const float g = c * sin(h) + 0.54371398;
	return vec3(Y, r, g);
}

vec3 rec2020_to_Ych(const vec3 rgb) {
	return Yrg_to_Ych(LMS_to_Yrg(XYZ_D65_to_LMS_2006_D65 * rec2020_to_xyz * rgb));
}

vec3 Ych_to_rec2020(const vec3 ych) {
	return xyz_to_rec2020 * LMS_2006_D65_to_XYZ_D65 * Yrg_to_LMS(Ych_to_Yrg(ych));
}

vec3 XYZ_to_xyY(vec3 xyz) {
	float norm = xyz.x + xyz.y + xyz.z;
	return vec3(xyz.xy / norm, xyz.y);
}

vec3 xyY_to_XYZ(vec3 xyY) {
	return vec3(xyY.x * xyY.z / xyY.y, xyY.z, (1 - xyY.x - xyY.y) * xyY.z / xyY.y);
}

vec3 // return adapted rec2020
cat16(vec3 xyz_in, vec3 rec2020_src, vec3 rec2020_dst)
{
  // these are the CAT16 M^{-1} and M matrices.
  // we use the standalone adaptation as proposed in
  // Smet and Ma, "Some concerns regarding the CAT16 chromatic adaptation transform",
  // Color Res Appl. 2020;45:172–177.
  // these are XYZ to cone-like
  const mat3 M16i = transpose(mat3(
       1.86206786, -1.01125463,  0.14918677,
       0.38752654,  0.62144744, -0.00897398,
      -0.01584150, -0.03412294,  1.04996444));
  const mat3 M16 = transpose(mat3(
       0.401288, 0.650173, -0.051461,
      -0.250268, 1.204414,  0.045854,
      -0.002079, 0.048952,  0.953127));

  const vec3 cl_src = M16 * rec2020_to_xyz * rec2020_src;
  const vec3 cl_dst = M16 * rec2020_to_xyz * rec2020_dst;
  vec3 cl = M16 * xyz_in;
  cl *= cl_dst / cl_src;
  return xyz_to_rec2020 * M16i * cl;
}
	

/* The following is darktable Uniform Color Space 2022
 * © Aurélien Pierre
 * https://eng.aurelienpierre.com/2022/02/color-saturation-control-for-the-21th-century/
 *
 * Use this space for color-grading in a perceptual framework.
 */

float Y_to_dt_UCS_L_star(const float Y) {
	// WARNING: L_star needs to be < 2.098883786377, meaning Y needs to be < 3.875766378407574e+19
	const float Y_hat = pow(Y, 0.631651345306265);
	return 2.098883786377 * Y_hat / (Y_hat + 1.12426773749357);
}

float dt_UCS_L_star_to_Y(const float L_star) {
	// WARNING: L_star needs to be < 2.098883786377, meaning Y needs to be < 3.875766378407574e+19
	return pow((1.12426773749357 * L_star / (2.098883786377 - L_star)), 1.5831518565279648);
}

vec2 xy_to_dt_UCS_UV(vec2 xy) {
	const mat3 M1 = mat3( // column vectors:
		-0.783941002840055,  0.745273540913283, 0.318707282433486,
		0.277512987809202, -0.205375866083878, 2.16743692732158,
		0.153836578598858, -0.165478376301988, 0.291320554395942);
	vec3 uvd = M1 * vec3(xy, 1.0);
	uvd.xy /= uvd.z;

	const vec2 factors     = vec2(1.39656225667, 1.4513954287 );
	const vec2 half_values = vec2(1.49217352929, 1.52488637914);
	vec2 UV_star = factors * uvd.xy / (abs(uvd.xy) + half_values);

	const mat2 M2 = mat2(-1.124983854323892, 1.86323315098672, - 0.980483721769325, + 1.971853092390862);
	return M2 * UV_star;
}

//  input :
//    * xyY in normalized CIE XYZ for the 2° 1931 observer adapted for D65
//    * L_white the lightness of white as dt UCS L* lightness
//    * cz = 1 for standard pre-print proofing conditions with average surround and n = 20 %
//            (background = middle grey, white = perfect diffuse white)
//  range : xy in [0; 1], Y normalized for perfect diffuse white = 1
vec3 xyY_to_dt_UCS_JCH(const vec3 xyY, const float L_white) {
	vec2 UV_star_prime = xy_to_dt_UCS_UV(xyY.xy);
	const float L_star = Y_to_dt_UCS_L_star(xyY.z);
	const float M2 = dot(UV_star_prime, UV_star_prime); // square of colorfulness M
	return vec3( // should be JCH[0] = powf(L_star / L_white), cz) but we treat only the case where cz = 1
		L_star / L_white,
		15.932993652962535 * pow(L_star, 0.6523997524738018) * pow(M2, 0.6007557017508491) / L_white,
		atan(UV_star_prime.y, UV_star_prime.x));
	}

vec3 dt_UCS_JCH_to_xyY(const vec3 JCH, const float L_white)
{
  const float L_star = JCH.x * L_white; // should be L_star = powf(JCH[0], 1.f / cz) * L_white but we treat only cz = 1
  const float M = pow(JCH.y * L_white / (15.932993652962535 * pow(L_star, 0.6523997524738018)), 0.8322850678616855);

  vec2 UV_star = M * vec2(cos(JCH.z), sin(JCH.z)); // uv*'
  const mat2 M1 = mat2(-5.037522385190711, 4.760029407436461, -2.504856328185843, 2.874012963239247); // col major
  UV_star = M1 * UV_star;

  const vec2 factors     = vec2(1.39656225667, 1.4513954287);
  const vec2 half_values = vec2(1.49217352929, 1.52488637914);
  vec2 UV = -half_values * UV_star / (abs(UV_star) - factors);

  const mat3 M2 = mat3( // given as column vectors
      vec3( 0.167171472114775,   -0.150959086409163,    0.940254742367256),
      vec3( 0.141299802443708,   -0.155185060382272,    1.000000000000000),
      vec3(-0.00801531300850582, -0.00843312433578007, -0.0256325967652889));
  vec3 xyD = M2 * vec3(UV, 1.0);
  return vec3(xyD.xy / xyD.z, dt_UCS_L_star_to_Y(L_star));
}

vec3 dt_UCS_JCH_to_HSB(const vec3 JCH) {
	vec3 HSB;
	HSB.z = JCH.x * (pow(JCH.y, 1.33654221029386) + 1.);
	return vec3(JCH.z, (HSB.z > 0.) ? JCH.y / HSB.z : 0., HSB.z);
}

vec3 dt_UCS_HSB_to_JCH(const vec3 HSB) {
	return vec3(HSB.z / (pow(HSB.y*HSB.z, 1.33654221029386) + 1.), HSB.y * HSB.z, HSB.x);
}

vec3 dt_UCS_JCH_to_HCB(const vec3 JCH) {
	return vec3(JCH.zy, JCH.x * (pow(JCH.y, 1.33654221029386) + 1.));
}

vec3 dt_UCS_HCB_to_JCH(const vec3 HCB) {
	return vec3(HCB.z / (pow(HCB.y, 1.33654221029386) + 1.), HCB.y, HCB.x);
}
