float fetch(ivec2 pos) {
    highp float v = texture(rawimg, pos).r;
    return v; 
}

vec3 debayer_linear(uvec2 pos) {
    /*
    variables a-i are the neighbour pixels (we are e)
    a b c
    d e f
    g h i
    */

    float a = texture(rawimg, ivec2(pos.x - 1, pos.y - 1)).r;
    float b = texture(rawimg, ivec2(pos.x    , pos.y - 1)).r;
    float c = texture(rawimg, ivec2(pos.x + 1, pos.y - 1)).r;
    float d = texture(rawimg, ivec2(pos.x - 1, pos.y    )).r;
    float e = texture(rawimg, ivec2(pos.x    , pos.y    )).r;
    float f = texture(rawimg, ivec2(pos.x + 1, pos.y    )).r;
    float g = texture(rawimg, ivec2(pos.x - 1, pos.y + 1)).r;
    float h = texture(rawimg, ivec2(pos.x    , pos.y + 1)).r;
    float i = texture(rawimg, ivec2(pos.x + 1, pos.y + 1)).r;

    vec3 red_pixel = vec3(
        e,
        (f + d + h + b) / 4.,
        (i + a + g + c) / 4.
    );
    vec3 blue_pixel = vec3(
        (i + a + g + c) / 4.,
        (f + d + h + b) / 4.,
        e
    );
    vec3 green_pixel_red_row = vec3(
        (d + f) / 2.,
        e,
        (b + h) / 2.
    );
    vec3 green_pixel_blue_row = vec3(
        (b + h) / 2.,
        e,
        (d + f) / 2.
    );

    float x_red = float((pos.x + 1) % 2);
    float y_red = float((pos.y + 1) % 2);

    vec3 rgb = (
        + red_pixel * x_red * y_red
        + blue_pixel * (1.0 - x_red) * (1.0 - y_red)
        + green_pixel_red_row * (1.0 - x_red) * y_red
        + green_pixel_blue_row * x_red * (1.0 - y_red)
    );
    return rgb;
}

vec3 debayer_mhc(uvec2 p) {
    ivec2 pos = ivec2(p);
    
    /* Load the sums of neighbours in the following pattern.
     * c is the current pixel; x, y, z, w are components of the vector v.
     *     x
     *   d y d
     * z w c w z
     *   d y d
     *     x
     */
    float c = fetch(pos);

    vec4 dvec = vec4(
        fetch(pos + ivec2(-1, -1)),
        fetch(pos + ivec2(-1, 1)),
        fetch(pos + ivec2(1, -1)),
        fetch(pos + ivec2(1, 1))
    );
    float d = dvec.x + dvec.y + dvec.z + dvec.w;
    
    vec4 v = vec4(  
        fetch(pos + ivec2(0, -2)),
        fetch(pos + ivec2(0, -1)),
        fetch(pos + ivec2(-2, 0)),
        fetch(pos + ivec2(-1, 0))
    );
    v += vec4(  
        fetch(pos + ivec2(0, 2)),
        fetch(pos + ivec2(0, 1)),
        fetch(pos + ivec2(2, 0)),
        fetch(pos + ivec2(1, 0))
    );

    /* There are four filter patterns, compute them as components of `pattern`.
     * cross  checker  theta   phi
     *   *       *       *       *
     *   *      * *     * *     *** 
     * *****   * * *   *****   * * *
     *   *      * *     * *     *** 
     *   *       *       *       *
     *
     * Component  Matches
     *   x        cross   (uses v.x, v.y, v.z, v.w)
     *   y        checker (uses d, v.x, v.z)
     *   z        theta   (uses d, v.x, v.z, v.w)
     *   w        phi     (uses d, v.x, v.y, v.z)
     */
    const vec4 c_coeffs = vec4( 4.0,  6.0,  5.0, 5.0) / 8.0;
    vec4 pattern = c_coeffs * c;

    const vec3 d_coeffs = vec3(2.0, -1.0, -1.0) / 8.0;
    pattern.yzw += d_coeffs * d;

    const vec4 xz_coeffs = vec4(-1.0, -1.5,  0.5, -1.0) / 8.0;
    pattern += xz_coeffs * v.x + xz_coeffs.xywz * v.z;
    
    const vec2 yw_coeffs = vec2(2.0,  4.0) / 8.0;
    pattern.xw += yw_coeffs * v.y; 
    pattern.xz += yw_coeffs * v.w; 
    
    bool x_red = (pos.x % 2) == X_RED;
    bool y_red = (pos.y % 2) == Y_RED;
    vec3 rgb = (
        x_red ? (y_red ? vec3(c, pattern.xy) : vec3(pattern.w, c, pattern.z) )
              : (y_red ? vec3(pattern.z, c, pattern.w) : vec3(pattern.yx, c) )
    );
    return rgb;
}

vec3 debayer_loss(uvec2 p) {
    ivec2 pos = ivec2(2*p);
    return vec3(
        fetch(pos),
        (fetch(pos + ivec2(1, 0)) + fetch(pos + ivec2(0, 1))) / 2,
        fetch(pos + ivec2(1, 1))
    );
}
