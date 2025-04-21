use std::sync::Arc;
use std::{array, cmp::min};

use foldhash::{HashMap, HashMapExt};
use serde::{Deserialize, Serialize};
use vulkano::command_buffer::BlitImageInfo;
use vulkano::{
    DeviceSize,
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferExecFuture, CommandBufferUsage,
        CopyBufferToImageInfo, CopyImageToBufferInfo, PrimaryCommandBufferAbstract,
        allocator::CommandBufferAllocator,
    },
    descriptor_set::{DescriptorSet, WriteDescriptorSet, allocator::DescriptorSetAllocator},
    device::Queue,
    format::Format,
    image::{
        Image, ImageCreateInfo, ImageType, ImageUsage,
        sampler::{Sampler, SamplerAddressMode, SamplerCreateInfo},
        view::ImageView,
    },
    memory::allocator::{
        AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter, StandardMemoryAllocator,
    },
    padded::Padded,
    pipeline::{
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo, compute::ComputePipelineCreateInfo,
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    shader::{ShaderModule, SpecializationConstant},
    sync::{
        GpuFuture,
        future::{FenceSignalFuture, NowFuture},
    },
};

use crate::color_utils::identity_mat;

type ComputeFuture = FenceSignalFuture<CommandBufferExecFuture<NowFuture>>;

/// Push constants for the compute shader
#[repr(C)]
#[derive(BufferContents, Clone, Copy, PartialEq, Deserialize, Serialize)]
pub struct PushConstantData {
    /// Camera color matrix
    pub cam_matrix: [Padded<[f32; 3], 4>; 3],
    /// Exposure
    pub exposure: f32,
    /// Global saturation
    pub saturation_global: f32,
    /// Shadows saturation
    pub saturation_shd: f32,
    /// Midtones saturation
    pub saturation_mid: f32,
    /// Highlights saturation
    pub saturation_hig: f32,
    /// White relative exposure
    pub white_re: f32,
    /// Black relative exposure
    pub black_re: f32,
    /// Contrast
    pub contrast: f32,
}

/// Compute shader specialization constants
#[derive(Clone, Copy, Deserialize, Serialize)]
pub struct Specialization {
    /// Coordinates of the first red pixel
    pub first_red: [u16; 2],
    /// Subtract this value from each pixel
    pub black_level: f32,
    /// Multiply each pixel by this value
    pub stretch: f32,
}

/// GPU compute operation
pub struct Compute {
    in_buffer: Option<Subbuffer<[u16]>>,
    in_image: Arc<ImageView>,
    out_images: [Arc<ImageView>; 2],
    compute_pipeline: Arc<ComputePipeline>,
    descriptor_sets: [Arc<DescriptorSet>; 2],
    queue: Arc<Queue>,
    memory_alloc: Arc<StandardMemoryAllocator>,
    command_buffer_alloc: Arc<dyn CommandBufferAllocator>,
    extent: [u32; 3],
    specialization: Specialization,
    future: Option<ComputeFuture>,
}

/// A context to run GPU computation in
#[derive(Clone)]
pub struct GpuContext {
    /// Compute shader
    pub shader: Arc<ShaderModule>,
    /// Vulkan queue
    pub queue: Arc<Queue>,
    /// Memory allocator
    pub memory_alloc: Arc<StandardMemoryAllocator>,
    /// Descriptor set allocator
    pub descriptor_set_alloc: Arc<dyn DescriptorSetAllocator>,
    /// Command buffer allocator
    pub command_buffer_alloc: Arc<dyn CommandBufferAllocator>,
}

type Extent = [u32; 2];

impl Compute {
    /// Create a new Compute struct for a video with the given specialization
    pub fn new(extent: Extent, spec: Specialization, gpu_context: GpuContext) -> Self {
        let GpuContext {
            shader,
            queue,
            memory_alloc,
            descriptor_set_alloc,
            command_buffer_alloc,
        } = gpu_context;
        let device = queue.device();
        let extent = [extent[0], extent[1], 1];

        let in_image = ImageView::new_default(
            Image::new(
                memory_alloc.clone(),
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: Format::R16_UNORM,
                    extent,
                    usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .unwrap(),
        )
        .unwrap();

        let out_images: [_; 2] = array::from_fn(|_| {
            ImageView::new_default(
                Image::new(
                    memory_alloc.clone(),
                    ImageCreateInfo {
                        image_type: ImageType::Dim2d,
                        format: Format::R16G16B16A16_UNORM,
                        extent,
                        usage: ImageUsage::SAMPLED | ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC,
                        ..Default::default()
                    },
                    AllocationCreateInfo::default(),
                )
                .unwrap(),
            )
            .unwrap()
        });

        let shader = shader.specialize(spec.make_const_map()).unwrap();
        let entry_point = shader.entry_point("main").unwrap();
        dbg!(entry_point.info().push_constant_requirements);
        let compute_pipeline = {
            let stage = PipelineShaderStageCreateInfo::new(entry_point);
            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();
            ComputePipeline::new(
                device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(stage, layout),
            )
            .unwrap()
        };

        let sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                unnormalized_coordinates: true,
                ..Default::default()
            },
        )
        .unwrap();

        let desc_layout = &compute_pipeline.layout().set_layouts()[0];
        let descriptor_sets: [_; 2] = array::from_fn(|i| {
            DescriptorSet::new(
                descriptor_set_alloc.clone(),
                desc_layout.clone(),
                [
                    WriteDescriptorSet::image_view_sampler(0, in_image.clone(), sampler.clone()),
                    WriteDescriptorSet::image_view(1, out_images[i].clone()),
                ],
                [],
            )
            .unwrap()
        });

        let extent = out_images[0].image().extent();
        Self {
            in_buffer: None,
            in_image,
            out_images,
            compute_pipeline,
            descriptor_sets,
            extent,
            memory_alloc,
            command_buffer_alloc,
            queue,
            specialization: spec,
            future: None,
        }
    }

    /// Get the image view of the output image with the index i (0 or 1)
    pub fn out_image(&self, i: usize) -> Arc<ImageView> {
        self.out_images[i].clone()
    }

    /// Run the computation with the push constants, optional input and output buffers
    /// (with the output buffer possibly containing the dimensions to resize the image)
    /// and the output image index
    pub fn process(
        &mut self,
        buffer: Option<Subbuffer<[u16]>>,
        index: usize,
        pc: PushConstantData,
        output: Option<(Subbuffer<[u8]>, Option<Extent>)>,
    ) {
        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_alloc.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        if let Some(buffer) = buffer {
            // buffer is behind a shared reference, so this is cheap
            self.in_buffer = Some(buffer.clone());
            builder
                .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
                    buffer,
                    self.in_image.image().clone(),
                ))
                .unwrap();
        }

        unsafe {
            builder
                .bind_pipeline_compute(self.compute_pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.compute_pipeline.layout().clone(),
                    0,
                    self.descriptor_sets[index].clone(),
                )
                .unwrap()
                .push_constants(self.compute_pipeline.layout().clone(), 0, pc)
                .unwrap()
                .dispatch([1 + self.extent[0] / 8, 1 + self.extent[1] / 8, 1])
                .unwrap();
        }

        if let Some((buffer, resize)) = output {
            let out_image = self.out_images[index].image().clone();
            let image = if let Some([w, h]) = resize {
                let blit_image = Image::new(
                    self.memory_alloc.clone(),
                    ImageCreateInfo {
                        image_type: ImageType::Dim2d,
                        format: Format::R16G16B16A16_UNORM,
                        extent: [w, h, 1],
                        usage: ImageUsage::SAMPLED
                            | ImageUsage::STORAGE
                            | ImageUsage::TRANSFER_SRC
                            | ImageUsage::TRANSFER_DST,
                        ..Default::default()
                    },
                    AllocationCreateInfo::default(),
                )
                .unwrap();
                let blit_info = BlitImageInfo::images(out_image, blit_image.clone());
                builder.blit_image(blit_info).unwrap();
                blit_image
            } else {
                out_image
            };
            builder
                .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(image, buffer))
                .unwrap();
        }

        self.future = Some(
            builder
                .build()
                .unwrap()
                .execute(self.queue.clone())
                .unwrap()
                .then_signal_fence_and_flush()
                .unwrap(),
        );
    }

    /// Is some computation being done?
    pub fn is_computing(&self) -> bool {
        self.future.is_some()
    }

    /// Has the computation finished?
    pub fn is_signaled(&self) -> bool {
        self.future
            .as_ref()
            .map(|f| f.is_signaled().unwrap())
            .unwrap_or(false)
    }

    /// Pause the computation
    pub fn pause(&mut self) {
        self.future = None;
    }

    /// Wait until the computation finished
    pub fn wait(&self) {
        if let Some(ref fut) = self.future {
            fut.wait(None).unwrap();
        }
    }

    /// Sample the pixel with coordinates `[x, y]` in range `0..=1',
    /// get the result in the camera RGB space.
    pub fn sample(&self, [x, y]: [f32; 2]) -> Option<[f32; 3]> {
        let buffer = self.in_buffer.as_ref()?;
        let (w, h) = (self.extent[0] as usize, self.extent[1] as usize);
        let (x, y) = ((x * w as f32) as usize, (y * h as f32) as usize);
        let (x, y) = (min(2 * (x / 2), w - 2), min(2 * (y / 2), h - 2));
        let cols = {
            let b = buffer.read().unwrap();
            [(0, 0), (0, 1), (1, 0), (1, 1)].map(|(i, j)| {
                ((b[(y + j) * w + x + i] as f32 / 65_535.0) - self.specialization.black_level)
                    * self.specialization.stretch
            })
        };
        let red = self.specialization.first_red[1] as usize * 2
            + self.specialization.first_red[0] as usize;
        let blue = 0b11 ^ red;
        let green = (0..4).filter(|i| *i != red && *i != blue);
        dbg!(Some([
            cols[red],
            green.map(|i| cols[i]).sum::<f32>() / 2.0,
            cols[blue],
        ]))
    }
}

/// Create an upload buffer
pub fn make_upload_buffer(
    width: usize,
    height: usize,
    alloc: Arc<dyn MemoryAllocator>,
) -> Subbuffer<[u16]> {
    Buffer::new_slice(
        alloc,
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        (width * height) as DeviceSize,
    )
    .unwrap()
}

/// Create an output buffer
pub fn make_output_buffer(
    width: usize,
    height: usize,
    alloc: Arc<dyn MemoryAllocator>,
) -> Subbuffer<[u8]> {
    Buffer::new_slice(
        alloc,
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        (width * height * 4 * 2) as DeviceSize,
    )
    .unwrap()
}

impl Specialization {
    fn make_const_map(&self) -> HashMap<u32, SpecializationConstant> {
        let mut map = HashMap::new();
        map.insert(0, SpecializationConstant::I32(self.first_red[0] as i32));
        map.insert(1, SpecializationConstant::I32(self.first_red[1] as i32));
        map.insert(2, SpecializationConstant::F32(self.black_level));
        map.insert(3, SpecializationConstant::F32(self.stretch));
        map
    }
}

impl Default for Specialization {
    fn default() -> Self {
        Specialization {
            first_red: [0, 0],
            black_level: 0.0,
            stretch: 1.0,
        }
    }
}

impl Default for PushConstantData {
    fn default() -> Self {
        Self {
            cam_matrix: identity_mat().map(Padded),
            exposure: 0.0,
            saturation_global: 0.0,
            saturation_shd: 0.0,
            saturation_mid: 0.0,
            saturation_hig: 0.0,
            white_re: 2.0,
            black_re: -6.0,
            contrast: 1.5,
        }
    }
}

impl Clone for Compute {
    fn clone(&self) -> Self {
        Compute {
            in_buffer: self.in_buffer.clone(),
            in_image: self.in_image.clone(),
            out_images: self.out_images.clone(),
            compute_pipeline: self.compute_pipeline.clone(),
            descriptor_sets: self.descriptor_sets.clone(),
            queue: self.queue.clone(),
            memory_alloc: self.memory_alloc.clone(),
            command_buffer_alloc: self.command_buffer_alloc.clone(),
            extent: self.extent,
            specialization: self.specialization,
            future: None,
        }
    }
}
