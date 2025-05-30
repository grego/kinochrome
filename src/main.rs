//! Color grade raw videos.
#![warn(missing_docs)]
/// Utilities for working with color spaces and corrections
pub mod color_utils;
/// Video encoding
pub mod encoding;
/// Interaction with vulkan compute
pub mod gpu_compute;
/// GUI layout
pub mod gui;
/// Parsers and decoders of various formats
pub mod import;
/// The render of egui to the Vulkan surface
pub mod renderer;
/// The application state and logic
pub mod state;

use encoding::EncodingDialog;
use gpu_compute::{Compute, GpuContext};
use import::{FocusPixelMap, parse_videos, read_frames};
use renderer::Renderer;
use state::State;

use std::error::Error;
use std::fs::File;
use std::time::Instant;

use std::collections::BTreeMap;
use std::io::{self, BufReader, ErrorKind, Read};
use std::sync::Arc;
use std::sync::mpsc::{channel, sync_channel};
use std::thread;

use egui_file_dialog::FileDialog;
use vulkano::{
    Validated, VulkanError, VulkanLibrary,
    command_buffer::allocator::{
        StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
        physical::PhysicalDeviceType,
    },
    format::Format,
    image::{Image, ImageUsage, view::ImageView},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::StandardMemoryAllocator,
    pipeline::graphics::viewport::Viewport,
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass},
    shader::{ShaderModule, ShaderModuleCreateInfo},
    swapchain::{
        Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo, acquire_next_image,
    },
    sync::{self, GpuFuture},
};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

const RECIPE_DIR: &str = "recipes";

/// The application
pub struct App {
    /// The application state
    state: State,
    /// The Vulkan device the application is running on
    instance: Arc<Instance>,
    /// The Vulkan device the application is running on
    device: Arc<Device>,
    /// The Vulkan queue for graphics rendering
    graphics_queue: Arc<Queue>,
    /// The vertex shader for the GUI renderer
    vertex_shader: Arc<ShaderModule>,
    /// The fragment shader for the GUI renderer
    fragment_shader: Arc<ShaderModule>,
    rcx: Option<RenderingContext>,
}

/// The context for the Vulkan rendering
pub struct RenderingContext {
    /// The opened window
    window: Arc<Window>,
    /// The swapchains for the images that are shown on the surface
    swapchain: Arc<Swapchain>,
    /// Recreate the swapchain
    recreate_swapchain: bool,
    /// Framebuffers
    framebuffers: Vec<Arc<Framebuffer>>,
    /// The are of the framebuffer that is rendered to
    viewport: Viewport,
    /// The future to be executed from the previous frame end
    previous_frame_end: Option<Box<dyn GpuFuture>>,
    /// The GUI renderer
    renderer: Renderer,
    /// How many frames do we want to redraw?
    redraw_frames: u8,
}

/// Initialize the application and start it
fn main() -> Result<(), impl Error> {
    let event_loop = EventLoop::new().unwrap();

    let library = VulkanLibrary::new().unwrap();
    let required_extensions = Surface::required_extensions(&event_loop).unwrap();
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            enabled_extensions: required_extensions,
            ..Default::default()
        },
    )
    .unwrap();

    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    };
    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    q.queue_flags.intersects(QueueFlags::GRAPHICS)
                        && p.presentation_support(i as u32, &event_loop)
                            .unwrap_or(false)
                })
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
            _ => 5,
        })
        .unwrap();

    println!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type,
    );

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: device_extensions,
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )
    .unwrap();
    let queue = queues.next().unwrap();
    let graphics_queue = queues.next().unwrap_or_else(|| queue.clone());

    let memory_alloc = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    let descriptor_set_alloc = Arc::new(StandardDescriptorSetAllocator::new(
        device.clone(),
        Default::default(),
    ));
    let command_buffer_alloc = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        Default::default(),
    ));

    let vspirv = read_file_as_u32s("shaders/vert.spv").unwrap();
    let vertex_shader =
        unsafe { ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(&vspirv)).unwrap() };
    let fspirv = read_file_as_u32s("shaders/frag.spv").unwrap();
    let fragment_shader =
        unsafe { ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(&fspirv)).unwrap() };
    let cspirv = read_file_as_u32s("shaders/comp.spv").unwrap();
    let shader =
        unsafe { ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(&cspirv)).unwrap() };
    let gpu_context = GpuContext {
        shader,
        queue,
        memory_alloc,
        descriptor_set_alloc,
        command_buffer_alloc,
    };

    let files: BTreeMap<String, _> = BTreeMap::new();
    let fpm: FocusPixelMap = Default::default();

    let cloned_fpm = fpm.clone();
    let (path_send, path_recv) = channel();
    let (vid_send, vid_recv) = channel();
    thread::spawn(|| parse_videos(path_recv, vid_send, cloned_fpm));

    let files_being_imported = if let Some(filename) = std::env::args().nth(1) {
        path_send.send((filename.into(), true)).unwrap();
        1
    } else {
        0
    };

    let cloned_alloc = gpu_context.memory_alloc.clone();
    let (cmd_send, cmd_recv) = channel();
    thread::spawn(|| read_frames(cloned_alloc, cmd_recv, true));
    let (_send, recv) = sync_channel(2);

    let extent = [1, 1];
    let ideal_frame_len = 1_000_000.0;
    let compute = Compute::new(extent, Default::default(), gpu_context.clone());

    let open_dialog = FileDialog::new()
        .add_file_filter(
            "Kinochrome projects",
            Arc::new(|path| {
                path.extension()
                    .unwrap_or_default()
                    .eq_ignore_ascii_case("kchrp")
            }),
        )
        .default_file_filter("Kinochrome projects")
        .default_file_name("project.kchrp");
    let import_dialog = FileDialog::new()
        .add_file_filter(
            "MLV files",
            Arc::new(|path| {
                path.extension()
                    .unwrap_or_default()
                    .eq_ignore_ascii_case("mlv")
            }),
        )
        .default_file_filter("MLV files");

    let (video_recipes, audio_recipes) = match encoding::read_recipes(RECIPE_DIR) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Unable to read recipes: {e}");
            Default::default()
        }
    };

    let state = State {
        gpu_context,
        files,
        filename: Default::default(),
        changed_file: None,
        extent,
        color_params: Default::default(),
        prev_color_params: Default::default(),
        undo_color_params: Default::default(),
        pc: Default::default(),
        undo_pc: Default::default(),
        compute,
        image_ids: None,
        fpm,

        cmd_send,
        recv,
        path_send,
        vid_recv,
        files_being_imported,

        paused: true,
        frames_len: 0,
        current_img: None,
        second_img: false,
        recompute: false,
        rewound_frame: false,
        frame_number: 0,
        encode_cdng: false,
        encoding_state: Default::default(),
        encoding_dialog: EncodingDialog {
            video_recipes,
            audio_recipes,
            ..Default::default()
        },

        first_start: Instant::now(),
        frame_start: Instant::now(),
        vid_frame_start: Instant::now(),
        frame_delta: 0.0,
        ideal_frame_len,

        open_dialog,
        import_dialog,
        about_dialog_shown: false,
        picker_mode: false,
        picked_point: None,
    };

    let mut app = App {
        instance,
        device,
        graphics_queue,
        vertex_shader,
        fragment_shader,
        state,
        rcx: None,
    };

    event_loop.run_app(&mut app)
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title(env!("CARGO_CRATE_NAME")))
                .unwrap(),
        );
        let surface = Surface::from_window(self.instance.clone(), window.clone()).unwrap();

        let (swapchain, images) = {
            let surface_capabilities = self
                .device
                .physical_device()
                .surface_capabilities(&surface, Default::default())
                .unwrap();
            let image_formats = self
                .device
                .physical_device()
                .surface_formats(&surface, Default::default())
                .unwrap();
            dbg!(&image_formats);
            let image_format = Format::B8G8R8A8_UNORM;

            Swapchain::new(
                self.device.clone(),
                surface.clone(),
                SwapchainCreateInfo {
                    min_image_count: surface_capabilities.min_image_count.max(2),
                    image_format,
                    image_extent: window.inner_size().into(),
                    image_usage: ImageUsage::COLOR_ATTACHMENT,
                    composite_alpha: surface_capabilities
                        .supported_composite_alpha
                        .into_iter()
                        .next()
                        .unwrap(),
                    ..Default::default()
                },
            )
            .unwrap()
        };

        let memory_alloc = Arc::new(StandardMemoryAllocator::new_default(self.device.clone()));
        let descriptor_set_alloc = Arc::new(StandardDescriptorSetAllocator::new(
            self.device.clone(),
            Default::default(),
        ));
        let command_buffer_alloc = Arc::new(StandardCommandBufferAllocator::new(
            self.device.clone(),
            StandardCommandBufferAllocatorCreateInfo {
                secondary_buffer_count: 32,
                ..Default::default()
            },
        ));

        let recreate_swapchain = false;
        let previous_frame_end: Option<Box<dyn GpuFuture>> =
            Some(sync::now(self.device.clone()).boxed());

        let mut viewport = Viewport {
            offset: [0.0, 0.0],
            extent: [0.0, 0.0],
            depth_range: 0.0..=1.0,
        };

        let mut renderer = Renderer::new(
            event_loop,
            surface.clone(),
            self.graphics_queue.clone(),
            images[0].format(),
            (self.vertex_shader.clone(), self.fragment_shader.clone()),
            (memory_alloc, descriptor_set_alloc, command_buffer_alloc),
        );
        let framebuffers =
            window_size_dependent_setup(&images, renderer.render_pass(), &mut viewport);

        // Register the output images to the GUI
        if !self.state.filename.is_empty() {
            self.state.update_current_file();
            self.state
                .change_file(&mut renderer, self.state.filename.clone());
        }

        self.rcx = Some(RenderingContext {
            window,
            swapchain,
            recreate_swapchain,
            framebuffers,
            viewport,
            previous_frame_end,
            renderer,
            redraw_frames: 2,
        })
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let rcx = self.rcx.as_mut().unwrap();

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                rcx.recreate_swapchain = true;
            }
            WindowEvent::RedrawRequested => {
                let window = rcx.window.as_ref();
                let image_extent: [u32; 2] = window.inner_size().into();

                if image_extent.contains(&0) {
                    return;
                }

                rcx.previous_frame_end.as_mut().unwrap().cleanup_finished();

                if rcx.recreate_swapchain {
                    let (new_swapchain, new_images) = rcx
                        .swapchain
                        .recreate(SwapchainCreateInfo {
                            image_extent,
                            ..rcx.swapchain.create_info()
                        })
                        .expect("failed to recreate swapchain");

                    rcx.swapchain = new_swapchain;
                    rcx.framebuffers = window_size_dependent_setup(
                        &new_images,
                        rcx.renderer.render_pass(),
                        &mut rcx.viewport,
                    );
                    rcx.recreate_swapchain = false;
                }

                let (image_index, suboptimal, acquire_future) = match acquire_next_image(
                    rcx.swapchain.clone(),
                    None,
                )
                .map_err(Validated::unwrap)
                {
                    Ok(r) => r,
                    Err(VulkanError::OutOfDate) => {
                        rcx.recreate_swapchain = true;
                        return;
                    }
                    Err(e) => panic!("failed to acquire next image: {e}"),
                };

                if suboptimal {
                    rcx.recreate_swapchain = true;
                }

                self.state.update(&mut rcx.renderer);
                let command_buffer = rcx
                    .renderer
                    .draw_on_image(rcx.framebuffers[image_index as usize].clone());

                let future = rcx
                    .previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(self.graphics_queue.clone(), command_buffer)
                    .unwrap()
                    .then_swapchain_present(
                        self.graphics_queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(
                            rcx.swapchain.clone(),
                            image_index,
                        ),
                    )
                    .then_signal_fence_and_flush();

                match future.map_err(Validated::unwrap) {
                    Ok(future) => {
                        match future.wait(None) {
                            Ok(x) => x,
                            Err(e) => println!("{e}"),
                        }
                        rcx.previous_frame_end = Some(future.boxed());
                    }
                    Err(VulkanError::OutOfDate) => {
                        rcx.recreate_swapchain = true;
                        rcx.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
                    }
                    Err(e) => {
                        println!("failed to flush future: {e}");
                        rcx.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
                    }
                }
            }
            event => {
                // Update Egui integration so the UI works!
                if rcx.renderer.update(&event) {
                    rcx.redraw_frames = 8;
                }
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let rcx = self.rcx.as_mut().unwrap();
        if self.state.redraw() {
            rcx.redraw_frames = 8;
        }
        if rcx.redraw_frames > 0 {
            rcx.redraw_frames -= 1;
            rcx.window.request_redraw();
        }
    }
}

/// This function is called once during initialization, then again whenever the window is resized.
fn window_size_dependent_setup(
    images: &[Arc<Image>],
    render_pass: Arc<RenderPass>,
    viewport: &mut Viewport,
) -> Vec<Arc<Framebuffer>> {
    let extent = images[0].extent();
    viewport.extent = [extent[0] as f32, extent[1] as f32];

    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}

/// Read a file as a sequence of little endian u32s
fn read_file_as_u32s(name: &str) -> Result<Vec<u32>, io::Error> {
    let file = File::open(name)?;
    let mut reader = BufReader::new(file);
    let mut words = Vec::new();
    let mut b = [0; 4];
    loop {
        match reader.read_exact(&mut b) {
            Ok(()) => words.push(u32::from_le_bytes(b)),
            Err(e) if e.kind() == ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }
    }
    Ok(words)
}
