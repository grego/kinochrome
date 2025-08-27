// Some code taken from egui vulkano inegration
// Copyright (c) 2021 Okko Hakola
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;

use bytemuck::AnyBitPattern;
use egui::{ClippedPrimitive, Rect, TexturesDelta, epaint::Primitive};
use foldhash::HashMap;
use vulkano::{
    DeviceSize, NonZeroDeviceSize,
    buffer::{
        Buffer, BufferCreateInfo, BufferUsage, Subbuffer,
        allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
    },
    command_buffer::{
        AutoCommandBufferBuilder, BufferImageCopy, CommandBufferInheritanceInfo,
        CommandBufferUsage, CopyBufferToImageInfo, PrimaryAutoCommandBuffer,
        PrimaryCommandBufferAbstract, RenderPassBeginInfo, SecondaryAutoCommandBuffer,
        SubpassBeginInfo, SubpassContents, allocator::CommandBufferAllocator,
    },
    descriptor_set::{
        DescriptorSet, WriteDescriptorSet, allocator::DescriptorSetAllocator,
        layout::DescriptorSetLayout,
    },
    device::Queue,
    format::Format,
    image::{
        Image, ImageAspects, ImageCreateInfo, ImageLayout, ImageSubresourceLayers, ImageType,
        ImageUsage, SampleCount,
        sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode},
        view::{ImageView, ImageViewCreateInfo},
    },
    memory::{
        DeviceAlignment,
        allocator::{
            AllocationCreateInfo, DeviceLayout, MemoryTypeFilter, StandardMemoryAllocator,
        },
    },
    pipeline::{
        DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{
                AttachmentBlend, BlendFactor, ColorBlendAttachmentState, ColorBlendState,
            },
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{
                Vertex, VertexBufferDescription, VertexDefinition, VertexInputRate,
                VertexMemberInfo,
            },
            viewport::{Scissor, Viewport, ViewportState},
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    render_pass::{Framebuffer, RenderPass, Subpass},
    shader::ShaderModule,
    swapchain::Surface,
    sync::GpuFuture,
};
use winit::event_loop::ActiveEventLoop;
use winit::window::Window;

const VERTICES_PER_QUAD: DeviceSize = 4;
const VERTEX_BUFFER_SIZE: DeviceSize = 1024 * 1024 * VERTICES_PER_QUAD;
const INDEX_BUFFER_SIZE: DeviceSize = 1024 * 1024 * 2;

/// The Vulkan renderer for egui
pub struct Renderer {
    egui_ctx: egui::Context,
    egui_winit: egui_winit::State,
    surface: Arc<Surface>,

    gfx_queue: Arc<Queue>,
    render_pass: Arc<RenderPass>,

    font_sampler: Arc<Sampler>,

    memory_alloc: Arc<StandardMemoryAllocator>,
    descriptor_set_alloc: Arc<dyn DescriptorSetAllocator>,
    command_buffer_alloc: Arc<dyn CommandBufferAllocator>,

    vertex_index_buffer_pool: SubbufferAllocator,
    pipeline: Arc<GraphicsPipeline>,
    subpass: Subpass,

    texture_desc_sets: HashMap<egui::TextureId, Arc<DescriptorSet>>,
    texture_images: HashMap<egui::TextureId, Arc<ImageView>>,
    next_native_tex_id: u64,
}

type VertexBuffer = Subbuffer<[egui::epaint::Vertex]>;
type IndexBuffer = Subbuffer<[u32]>;

/// Should match vertex definition of egui
#[repr(C)]
#[derive(AnyBitPattern, Clone, Copy)]
struct EguiVertex {
    position: [f32; 2],
    tex_coords: [f32; 2],
    color: [u8; 4],
}

#[repr(C)]
#[derive(AnyBitPattern, Clone, Copy)]
struct PushConstants {
    screen_size: [f32; 2],
    needs_color_convert: i32,
}

impl Renderer {
    /// Creates new Egui to Vulkano integration by setting the necessary parameters
    /// This is to be called once we have access to winit window surface
    /// and gfx queue. Created with this, the renderer will own a render pass which is useful to e.g. place your render pass' images
    /// onto egui windows
    pub fn new(
        event_loop: &ActiveEventLoop,
        surface: Arc<Surface>,
        gfx_queue: Arc<Queue>,
        output_format: Format,
        (vs, fs): (Arc<ShaderModule>, Arc<ShaderModule>),
        (memory_alloc, descriptor_set_alloc, command_buffer_alloc): (
            Arc<StandardMemoryAllocator>,
            Arc<dyn DescriptorSetAllocator>,
            Arc<dyn CommandBufferAllocator>,
        ),
    ) -> Renderer {
        // Create Gui render pass with just depth and final color
        let render_pass = vulkano::single_pass_renderpass!(gfx_queue.device().clone(),
            attachments: {
                final_color: {
                    format: output_format,
                    samples: SampleCount::Sample1,
                    load_op: Load,
                    store_op: Store,
                }
            },
            pass: {
                    color: [final_color],
                    depth_stencil: {}
            }
        )
        .unwrap();
        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

        let vertex_index_buffer_pool = SubbufferAllocator::new(
            memory_alloc.clone(),
            SubbufferAllocatorCreateInfo {
                arena_size: INDEX_BUFFER_SIZE + VERTEX_BUFFER_SIZE,
                buffer_usage: BufferUsage::INDEX_BUFFER | BufferUsage::VERTEX_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );
        let pipeline = Self::create_pipeline(gfx_queue.clone(), vs, fs, subpass.clone());
        let font_sampler = Sampler::new(
            gfx_queue.device().clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                mipmap_mode: SamplerMipmapMode::Linear,
                ..Default::default()
            },
        )
        .unwrap();

        let max_texture_side = gfx_queue
            .device()
            .physical_device()
            .properties()
            .max_image_dimension2_d as usize;
        let egui_ctx: egui::Context = Default::default();
        let egui_winit = egui_winit::State::new(
            egui_ctx.clone(),
            egui_ctx.viewport_id(),
            event_loop,
            Some(surface_window(&surface).scale_factor() as f32),
            None,
            Some(max_texture_side),
        );
        Renderer {
            egui_ctx,
            egui_winit,
            surface,
            gfx_queue,
            render_pass,
            vertex_index_buffer_pool,
            pipeline,
            subpass,
            texture_desc_sets: HashMap::default(),
            texture_images: HashMap::default(),
            next_native_tex_id: 0,
            font_sampler,
            memory_alloc,
            descriptor_set_alloc,
            command_buffer_alloc,
        }
    }

    /// Returns the pixels per point of the window of this gui.
    fn pixels_per_point(&self) -> f32 {
        egui_winit::pixels_per_point(&self.egui_ctx, surface_window(&self.surface))
    }

    /// Updates context state by winit window event.
    /// Returns `true` if we need to redraw because of this event
    pub fn update(&mut self, winit_event: &winit::event::WindowEvent) -> bool {
        self.egui_winit
            .on_window_event(surface_window(&self.surface), winit_event)
            .repaint
    }

    /// Begins Egui frame. This must be called before draw, and after `update` (winit event).
    pub fn begin_frame(&mut self) {
        let raw_input = self
            .egui_winit
            .take_egui_input(surface_window(&self.surface));
        self.egui_ctx.begin_pass(raw_input);
    }

    fn end_frame(&mut self) -> (Vec<ClippedPrimitive>, TexturesDelta) {
        let egui::FullOutput {
            platform_output,
            textures_delta,
            shapes,
            ..
        } = self.egui_ctx.end_pass();

        self.egui_winit
            .handle_platform_output(surface_window(&self.surface), platform_output);
        let clipped_meshes = self.egui_ctx.tessellate(shapes, self.pixels_per_point());
        (clipped_meshes, textures_delta)
    }

    /// Access egui's context (which can be used to e.g. set fonts, visuals etc)
    pub fn context(&self) -> egui::Context {
        self.egui_ctx.clone()
    }

    /// Get the GUI render pass
    pub fn render_pass(&self) -> Arc<RenderPass> {
        self.render_pass.clone()
    }

    fn create_pipeline(
        gfx_queue: Arc<Queue>,
        vs: Arc<ShaderModule>,
        fs: Arc<ShaderModule>,
        subpass: Subpass,
    ) -> Arc<GraphicsPipeline> {
        let mut blend = AttachmentBlend::alpha();
        blend.src_color_blend_factor = BlendFactor::One;
        blend.src_alpha_blend_factor = BlendFactor::OneMinusDstAlpha;
        blend.dst_alpha_blend_factor = BlendFactor::One;
        let blend_state = ColorBlendState {
            attachments: vec![ColorBlendAttachmentState {
                blend: Some(blend),
                ..Default::default()
            }],
            ..ColorBlendState::default()
        };

        let vs = vs.entry_point("main").unwrap();
        let fs = fs.entry_point("main").unwrap();

        let vertex_input_state = Some(EguiVertex::per_vertex().definition(&vs).unwrap());

        let stages = [
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs),
        ];

        let layout = PipelineLayout::new(
            gfx_queue.device().clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(gfx_queue.device().clone())
                .unwrap(),
        )
        .unwrap();

        GraphicsPipeline::new(
            gfx_queue.device().clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state,
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState::default()),
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState {
                    rasterization_samples: subpass.num_samples().unwrap_or(SampleCount::Sample1),
                    ..Default::default()
                }),
                color_blend_state: Some(blend_state),
                dynamic_state: [DynamicState::Viewport, DynamicState::Scissor]
                    .into_iter()
                    .collect(),
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
        .unwrap()
    }

    /// Creates a descriptor set for images
    fn sampled_image_desc_set(
        &self,
        layout: &Arc<DescriptorSetLayout>,
        image: Arc<ImageView>,
        sampler: Arc<Sampler>,
    ) -> Arc<DescriptorSet> {
        DescriptorSet::new(
            self.descriptor_set_alloc.clone(),
            layout.clone(),
            [WriteDescriptorSet::image_view_sampler(0, image, sampler)],
            [],
        )
        .unwrap()
    }

    /// Registers a user texture. User texture needs to be unregistered when it is no longer needed
    pub fn register_image(
        &mut self,
        image: Arc<ImageView>,
        sampler_create_info: SamplerCreateInfo,
    ) -> egui::TextureId {
        let layout = self.pipeline.layout().set_layouts().first().unwrap();
        let sampler = Sampler::new(self.gfx_queue.device().clone(), sampler_create_info).unwrap();
        let desc_set = self.sampled_image_desc_set(layout, image.clone(), sampler);
        let id = egui::TextureId::User(self.next_native_tex_id);
        self.next_native_tex_id += 1;
        self.texture_desc_sets.insert(id, desc_set);
        self.texture_images.insert(id, image);
        id
    }

    /// Unregister user texture.
    pub fn unregister_image(&mut self, texture_id: egui::TextureId) {
        self.texture_desc_sets.remove(&texture_id);
        self.texture_images.remove(&texture_id);
    }

    fn image_size_bytes(&self, delta: &egui::epaint::ImageDelta) -> usize {
        match &delta.image {
            egui::ImageData::Color(c) => {
                // Always four bytes per pixel for sRGBA
                c.width() * c.height() * 4
            }
        }
    }

    /// Write a single texture delta using the provided staging region and commandbuffer
    fn update_texture_within(
        &mut self,
        id: egui::TextureId,
        delta: &egui::epaint::ImageDelta,
        stage: Subbuffer<[u8]>,
        mapped_stage: &mut [u8],
        cbb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) {
        // Extract pixel data from egui, writing into our region of the stage buffer.
        let format = match &delta.image {
            egui::ImageData::Color(image) => {
                assert_eq!(
                    image.width() * image.height(),
                    image.pixels.len(),
                    "Mismatch between texture size and texel count"
                );
                let bytes = image.pixels.iter().flat_map(|color| color.to_array());
                mapped_stage
                    .iter_mut()
                    .zip(bytes)
                    .for_each(|(into, from)| *into = from);
                Format::R8G8B8A8_SRGB
            }
        };

        // Copy texture data to existing image if delta pos exists (e.g. font changed)
        if let Some(pos) = delta.pos {
            let Some(existing_image) = self.texture_images.get(&id) else {
                // Egui wants us to update this texture but we don't have it to begin with!
                panic!("attempt to write into non-existing image");
            };
            // Make sure delta image type and destination image type match.
            assert_eq!(existing_image.format(), format);

            // Defer upload of data
            cbb.copy_buffer_to_image(CopyBufferToImageInfo {
                regions: [BufferImageCopy {
                    // Buffer offsets are derived
                    image_offset: [pos[0] as u32, pos[1] as u32, 0],
                    image_extent: [delta.image.width() as u32, delta.image.height() as u32, 1],
                    // Always use the whole image (no arrays or mips are performed)
                    image_subresource: ImageSubresourceLayers {
                        aspects: ImageAspects::COLOR,
                        mip_level: 0,
                        array_layers: 0..1,
                    },
                    ..Default::default()
                }]
                .into(),
                ..CopyBufferToImageInfo::buffer_image(stage, existing_image.image().clone())
            })
            .unwrap();
        } else {
            // Otherwise save the newly created image
            let img = {
                let extent = [delta.image.width() as u32, delta.image.height() as u32, 1];
                Image::new(
                    self.memory_alloc.clone(),
                    ImageCreateInfo {
                        image_type: ImageType::Dim2d,
                        format,
                        extent,
                        usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                        initial_layout: ImageLayout::Undefined,
                        ..Default::default()
                    },
                    AllocationCreateInfo::default(),
                )
                .unwrap()
            };
            // Defer upload of data
            cbb.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(stage, img.clone()))
                .unwrap();
            let view = ImageView::new(
                img.clone(),
                ImageViewCreateInfo {
                    ..ImageViewCreateInfo::from_image(&img)
                },
            )
            .unwrap();
            // Create a descriptor for it
            let layout = self.pipeline.layout().set_layouts().first().unwrap();
            let desc_set =
                self.sampled_image_desc_set(layout, view.clone(), self.font_sampler.clone());
            // Save!
            self.texture_desc_sets.insert(id, desc_set);
            self.texture_images.insert(id, view);
        };
    }

    /// Write the entire texture delta for this frame.
    fn update_textures(&mut self, sets: &[(egui::TextureId, egui::epaint::ImageDelta)]) {
        // Allocate enough memory to upload every delta at once.
        let total_size_bytes = sets
            .iter()
            .map(|(_, set)| self.image_size_bytes(set))
            .sum::<usize>()
            * 4;
        // Infallible - unless we're on a 128 bit machine? :P
        let total_size_bytes = u64::try_from(total_size_bytes).unwrap();
        let Ok(total_size_bytes) = vulkano::NonZeroDeviceSize::try_from(total_size_bytes) else {
            // Nothing to upload!
            return;
        };
        let buffer = Buffer::new(
            self.memory_alloc.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            // Bytes, align of one, infallible.
            DeviceLayout::new(total_size_bytes, DeviceAlignment::MIN).unwrap(),
        )
        .unwrap();
        let buffer = Subbuffer::new(buffer);

        // Shared command buffer for every upload in this batch.
        let mut cbb = AutoCommandBufferBuilder::primary(
            self.command_buffer_alloc.clone(),
            self.gfx_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        {
            // Scoped to keep writer lock bounded
            // Should be infallible - Just made the buffer so it's exclusive, and we have host access to it.
            let mut writer = buffer.write().unwrap();

            // Keep track of where to write the next image to into the staging buffer.
            let mut past_buffer_end = 0usize;

            for (id, delta) in sets {
                let image_size_bytes = self.image_size_bytes(delta);
                let range = past_buffer_end..(image_size_bytes + past_buffer_end);

                // Bump for next loop
                past_buffer_end += image_size_bytes;

                // Represents the same memory in two ways. Writable memmap, and gpu-side description.
                let stage = buffer.clone().slice(range.start as u64..range.end as u64);
                let mapped_stage = &mut writer[range];

                self.update_texture_within(*id, delta, stage, mapped_stage, &mut cbb);
            }
        }

        // Execute every upload at once and await:
        let command_buffer = cbb.build().unwrap();
        // Executing on the graphics queue not only since it's what we have, but
        // we must guarantee a transfer granularity of [1,1,x] which graphics queue is required to have.
        command_buffer
            .execute(self.gfx_queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
    }

    fn get_rect_scissor(
        &self,
        scale_factor: f32,
        framebuffer_dimensions: [u32; 2],
        rect: Rect,
    ) -> Scissor {
        let min = rect.min;
        let min = egui::Pos2 {
            x: min.x * scale_factor,
            y: min.y * scale_factor,
        };
        let min = egui::Pos2 {
            x: min.x.clamp(0.0, framebuffer_dimensions[0] as f32),
            y: min.y.clamp(0.0, framebuffer_dimensions[1] as f32),
        };
        let max = rect.max;
        let max = egui::Pos2 {
            x: max.x * scale_factor,
            y: max.y * scale_factor,
        };
        let max = egui::Pos2 {
            x: max.x.clamp(min.x, framebuffer_dimensions[0] as f32),
            y: max.y.clamp(min.y, framebuffer_dimensions[1] as f32),
        };
        Scissor {
            offset: [min.x.round() as u32, min.y.round() as u32],
            extent: [
                (max.x.round() - min.x) as u32,
                (max.y.round() - min.y) as u32,
            ],
        }
    }

    fn create_secondary_command_buffer_builder(
        &self,
    ) -> AutoCommandBufferBuilder<SecondaryAutoCommandBuffer> {
        AutoCommandBufferBuilder::secondary(
            self.command_buffer_alloc.clone(),
            self.gfx_queue.queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
            CommandBufferInheritanceInfo {
                render_pass: Some(self.subpass.clone().into()),
                ..Default::default()
            },
        )
        .unwrap()
    }

    /// Executes our draw commands on the final image and returns a command buffer to run
    pub fn draw_on_image(
        &mut self,
        framebuffer: Arc<Framebuffer>,
    ) -> Arc<PrimaryAutoCommandBuffer> {
        let (clipped_meshes, textures_delta) = self.end_frame();
        let scale_factor = self.pixels_per_point();
        self.update_textures(&textures_delta.set);

        // Get dimensions
        let img_dims = framebuffer.extent();
        let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_alloc.clone(),
            self.gfx_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        // Add clear values here for attachments and begin render pass
        command_buffer_builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![None],
                    ..RenderPassBeginInfo::framebuffer(framebuffer)
                },
                SubpassBeginInfo {
                    contents: SubpassContents::SecondaryCommandBuffers,
                    ..SubpassBeginInfo::default()
                },
            )
            .unwrap();

        let mut builder = self.create_secondary_command_buffer_builder();
        self.draw_egui(
            scale_factor,
            &clipped_meshes,
            [img_dims[0], img_dims[1]],
            &mut builder,
        );
        // Execute draw commands
        let command_buffer = builder.build().unwrap();
        command_buffer_builder
            .execute_commands(command_buffer)
            .unwrap()
            .end_render_pass(Default::default())
            .unwrap();
        let command_buffer = command_buffer_builder.build().unwrap();

        for &id in &textures_delta.free {
            self.unregister_image(id);
        }

        command_buffer
    }

    /// Uploads all meshes in bulk. They will be available in the same order, packed.
    /// None if no vertices or no indices.
    fn upload_meshes(
        &mut self,
        clipped_meshes: &[ClippedPrimitive],
    ) -> Option<(VertexBuffer, IndexBuffer)> {
        use egui::epaint::Vertex;
        type Index = u32;
        const VERTEX_ALIGN: DeviceAlignment = DeviceAlignment::of::<Vertex>();
        const INDEX_ALIGN: DeviceAlignment = DeviceAlignment::of::<Index>();

        // Iterator over only the meshes, no user callbacks.
        let meshes = clipped_meshes
            .iter()
            .filter_map(|mesh| match &mesh.primitive {
                Primitive::Mesh(m) => Some(m),
                _ => None,
            });

        // Calculate counts of each mesh, and total bytes for combined data
        let (total_vertices, total_size_bytes) = {
            let mut total_vertices = 0;
            let mut total_indices = 0;

            for mesh in meshes.clone() {
                total_vertices += mesh.vertices.len();
                total_indices += mesh.indices.len();
            }
            if total_indices == 0 || total_vertices == 0 {
                return None;
            }

            let total_size_bytes = total_vertices * std::mem::size_of::<Vertex>()
                + total_indices * std::mem::size_of::<Index>();
            (
                total_vertices,
                // Infallible! Checked above.
                NonZeroDeviceSize::new(u64::try_from(total_size_bytes).unwrap()).unwrap(),
            )
        };

        // Allocate a buffer which can hold both packed arrays:
        let layout = DeviceLayout::new(total_size_bytes, VERTEX_ALIGN.max(INDEX_ALIGN)).unwrap();
        let buffer = self.vertex_index_buffer_pool.allocate(layout).unwrap();

        // We must put the items with stricter align *first* in the packed buffer.
        // Correct at time of writing, but assert in case that changes.
        assert!(VERTEX_ALIGN >= INDEX_ALIGN);
        let (vertices, indices) = {
            let partition_bytes = total_vertices as u64 * std::mem::size_of::<Vertex>() as u64;
            (
                // Slice the start as vertices
                buffer
                    .clone()
                    .slice(..partition_bytes)
                    .reinterpret::<[Vertex]>(),
                // Take the rest, reinterpret as indices.
                buffer.slice(partition_bytes..).reinterpret::<[Index]>(),
            )
        };

        // We have to upload in two mapping steps to avoid trivial but ugly unsafe.
        {
            let mut vertex_write = vertices.write().unwrap();
            vertex_write
                .iter_mut()
                .zip(meshes.clone().flat_map(|m| &m.vertices).copied())
                .for_each(|(into, from)| *into = from);
        }
        {
            let mut index_write = indices.write().unwrap();
            index_write
                .iter_mut()
                .zip(meshes.flat_map(|m| &m.indices).copied())
                .for_each(|(into, from)| *into = from);
        }

        Some((vertices, indices))
    }

    fn draw_egui(
        &mut self,
        scale_factor: f32,
        clipped_meshes: &[ClippedPrimitive],
        framebuffer_dimensions: [u32; 2],
        builder: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>,
    ) {
        let mut push_constants = PushConstants {
            screen_size: [
                framebuffer_dimensions[0] as f32 / scale_factor,
                framebuffer_dimensions[1] as f32 / scale_factor,
            ],
            needs_color_convert: 1,
        };

        let mesh_buffers = self.upload_meshes(clipped_meshes);

        // Current position of renderbuffers, advances as meshes are consumed.
        let mut vertex_cursor = 0;
        let mut index_cursor = 0;
        // Some of our state is immutable and only changes
        // if a user callback thrashes it, rebind all when this is set:
        let mut needs_full_rebind = true;

        let mut was_user_image = true;
        // Track resources that change from call-to-call.
        // egui already makes the optimization that draws with identical resources are merged into one,
        // so every mesh changes usually one or possibly both of these.
        let mut current_rect = None;
        let mut current_texture = None;

        for ClippedPrimitive {
            clip_rect,
            primitive,
        } in clipped_meshes
        {
            match primitive {
                Primitive::Mesh(mesh) => {
                    // Nothing to draw if we don't have vertices & indices
                    if mesh.vertices.is_empty() || mesh.indices.is_empty() {
                        // Consume the mesh and skip it.
                        index_cursor += mesh.indices.len() as u32;
                        vertex_cursor += mesh.vertices.len() as u32;
                        continue;
                    }

                    if let egui::TextureId::User(_) = mesh.texture_id {
                        push_constants.needs_color_convert = 0;
                        needs_full_rebind = true;
                        was_user_image = true;
                    } else if was_user_image {
                        push_constants.needs_color_convert = 1;
                        was_user_image = false;
                        needs_full_rebind = true;
                    }
                    // Reset overall state, if needed.
                    // Only happens on first mesh, and after a user callback which does unknowable
                    // things to the command buffer's state.
                    if needs_full_rebind {
                        needs_full_rebind = false;

                        // Bind combined meshes.
                        let Some((vertices, indices)) = mesh_buffers.clone() else {
                            // Only None if there are no mesh calls, but here we are in a mesh call!
                            unreachable!()
                        };

                        builder
                            .bind_pipeline_graphics(self.pipeline.clone())
                            .unwrap()
                            .bind_index_buffer(indices)
                            .unwrap()
                            .bind_vertex_buffers(0, [vertices])
                            .unwrap()
                            .set_viewport(
                                0,
                                [Viewport {
                                    offset: [0.0, 0.0],
                                    extent: [
                                        framebuffer_dimensions[0] as f32,
                                        framebuffer_dimensions[1] as f32,
                                    ],
                                    depth_range: 0.0..=1.0,
                                }]
                                .into_iter()
                                .collect(),
                            )
                            .unwrap()
                            .push_constants(self.pipeline.layout().clone(), 0, push_constants)
                            .unwrap();
                    }
                    // Find and bind image, if different.
                    if current_texture != Some(mesh.texture_id) {
                        if !self.texture_desc_sets.contains_key(&mesh.texture_id) {
                            eprintln!("This texture no longer exists {:?}", mesh.texture_id);
                            continue;
                        }
                        current_texture = Some(mesh.texture_id);

                        let desc_set = self.texture_desc_sets.get(&mesh.texture_id).unwrap();

                        builder
                            .bind_descriptor_sets(
                                PipelineBindPoint::Graphics,
                                self.pipeline.layout().clone(),
                                0,
                                desc_set.clone(),
                            )
                            .unwrap();
                    };
                    // Calculate and set scissor, if different
                    if current_rect != Some(*clip_rect) {
                        current_rect = Some(*clip_rect);
                        let new_scissor =
                            self.get_rect_scissor(scale_factor, framebuffer_dimensions, *clip_rect);

                        builder
                            .set_scissor(0, [new_scissor].into_iter().collect())
                            .unwrap();
                    }

                    // All set up to draw!
                    unsafe {
                        builder
                            .draw_indexed(
                                mesh.indices.len() as u32,
                                1,
                                index_cursor,
                                vertex_cursor as i32,
                                0,
                            )
                            .unwrap();
                    }

                    // Consume this mesh for next iteration
                    index_cursor += mesh.indices.len() as u32;
                    vertex_cursor += mesh.vertices.len() as u32;
                }
                Primitive::Callback(_) => {
                    eprintln!("Warning: render callbacks are not supported.");
                }
            }
        }
    }
}

// Helper to retrieve Window from surface object
fn surface_window(surface: &Surface) -> &Window {
    surface.object().unwrap().downcast_ref::<Window>().unwrap()
}

unsafe impl Vertex for EguiVertex {
    fn per_vertex() -> VertexBufferDescription {
        use std::collections::HashMap;
        use std::mem;

        let mut offset: u32 = 0;
        let mut members = HashMap::default();

        let field_align = ::std::mem::align_of::<[f32; 2]>() as u32;
        offset = (offset + field_align - 1) & !(field_align - 1);
        members.insert(
            "position".to_string(),
            VertexMemberInfo {
                offset,
                format: Format::R32G32_SFLOAT,
                num_elements: 1,
                stride: 0,
            },
        );
        offset += mem::size_of::<[f32; 2]>() as u32;

        offset = (offset + field_align - 1) & !(field_align - 1);
        members.insert(
            "tex_coords".to_string(),
            VertexMemberInfo {
                offset,
                format: Format::R32G32_SFLOAT,
                num_elements: 1,
                stride: 0,
            },
        );
        offset += mem::size_of::<[f32; 2]>() as u32;

        let field_align = ::std::mem::align_of::<[u8; 4]>() as u32;
        offset = (offset + field_align - 1) & !(field_align - 1);
        members.insert(
            "color".to_string(),
            VertexMemberInfo {
                offset,
                format: Format::R8G8B8A8_UNORM,
                num_elements: 1,
                stride: 0,
            },
        );

        VertexBufferDescription {
            members,
            stride: mem::size_of::<Self>() as u32,
            input_rate: VertexInputRate::Vertex,
        }
    }

    fn per_instance() -> VertexBufferDescription {
        Self::per_vertex().per_instance()
    }

    fn per_instance_with_divisor(divisor: u32) -> VertexBufferDescription {
        Self::per_vertex().per_instance_with_divisor(divisor)
    }
}
