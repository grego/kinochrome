SHADERC = glslc

SHD_HEADERS=$(wildcard shaders/*.h)

all: shaders/comp.spv shaders/vert.spv shaders/frag.spv

shaders/comp.spv: shaders/comp.glsl $(SHD_HEADERS)
	@printf 'SHADERC\t%s\n' '$@'
	@$(SHADERC) -fshader-stage=comp --target-env=vulkan -O $< -o $@

shaders/vert.spv: shaders/vert.glsl
	@printf 'SHADERC\t%s\n' '$@'
	@$(SHADERC) -fshader-stage=vert $< -o $@

shaders/frag.spv: shaders/frag.glsl
	@printf 'SHADERC\t%s\n' '$@'
	@$(SHADERC) -fshader-stage=frag --target-env=vulkan -O $< -o $@
