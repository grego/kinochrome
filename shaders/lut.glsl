#version 450
#extension GL_EXT_shader_explicit_arithmetic_types: enable

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer RawImage {
	uint16_t rawimg[];
};

layout(set = 0, binding = 1) readonly buffer Lut {
	uint16_t lut[4096];
};

void main() {
	uint p = gl_GlobalInvocationID.x;
	rawimg[p] = lut[rawimg[p]];
}
