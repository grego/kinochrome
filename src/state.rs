use crate::color_utils::{ColorParams, Illuminant};
use crate::encoding::{self, EncodingDialog, EncodingState};
use crate::gpu_compute::{Compute, GpuContext, PushConstantData};
use crate::gui;
use crate::import::{FocusPixelMap, VideoCommand, VideoFile, load_focus_pixels};
use crate::renderer::Renderer;

use egui::util::undoer::Undoer;
use egui::{Pos2, TextureId};
use egui_file_dialog::{DialogMode, FileDialog};
use vulkano::buffer::Subbuffer;
use vulkano::image::sampler::{Filter, SamplerCreateInfo};

use std::collections::BTreeMap;
use std::fs::File;
use std::io;
use std::ops::Range;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;
use std::{array, env};

/// Application state
pub struct State {
    /// GPU context
    pub gpu_context: GpuContext,
    /// All imported files
    pub files: BTreeMap<String, VideoFile>,
    /// Filename of the currently edited file
    pub filename: String,
    /// Name of a new file about to be edited
    pub changed_file: Option<String>,
    /// Current video extent
    pub extent: [u32; 2],
    /// Current video color params
    pub color_params: ColorParams,
    /// Previous color params
    pub prev_color_params: ColorParams,
    /// Undoer for the color parameters
    pub undo_color_params: Undoer<ColorParams>,
    /// Current push constant data
    pub pc: PushConstantData,
    /// Push constant data from the previous frame.
    /// Needed to determine whether a redraw is necessary
    pub prev_pc: PushConstantData,
    /// Previous push constant data
    pub undo_pc: Undoer<PushConstantData>,
    /// GPU compute operation
    pub compute: Compute,
    /// Do we want to recompute?
    pub recompute: bool,
    /// egui ids of video images
    pub image_ids: Option<[TextureId; 2]>,
    /// A map of focus pixels for each camera and resolution
    pub fpm: FocusPixelMap,

    /// Sending half of a channel to the decoding thread
    pub cmd_send: Sender<VideoCommand>,
    /// Receiver of decoded images
    pub recv: Receiver<(Subbuffer<[u16]>, usize)>,
    /// Sending of paths of videos to parse, the boolean parameter determines whether the video
    /// should be focused on afterwards.
    pub path_send: Sender<(PathBuf, bool)>,
    /// Receiver of the parsed videos, the boolean parameter determines whether the video
    /// should be focused on afterwards.
    pub vid_recv: Receiver<(io::Result<VideoFile>, String, bool)>,
    /// The number of files being imported now
    pub files_being_imported: usize,

    /// Is the video paused?
    pub paused: bool,
    /// Number of the frames
    pub frames_len: usize,
    /// Current image being displayed
    pub current_img: Option<TextureId>,
    /// Is the second image being displayed?
    pub second_img: bool,
    /// Number of the current frame
    pub frame_number: usize,
    /// Was the frame rewound?
    pub rewound_frame: bool,
    /// The first and the last frame the video should be trimmed to.
    pub trim: Range<usize>,

    /// Start of this video
    pub first_start: Instant,
    /// Start of the previous frame
    pub frame_start: Instant,
    /// Start of the previous frame
    pub vid_frame_start: Instant,
    /// Time difference for the previous frame, in microseconds
    pub frame_delta: f32,
    /// Ideal length for the desired FPS, in microseconds
    pub ideal_frame_len: f32,

    /// Are the picked files going to be encoded as CinemaDNG or through ffmpeg?
    pub encode_cdng: bool,
    /// Encoding state
    pub encoding_state: Arc<Mutex<EncodingState>>,
    /// Encoding dialog
    pub encoding_dialog: EncodingDialog,

    /// The dialog for opening and saving projects
    pub open_dialog: FileDialog,
    /// The dialog for importing files
    pub import_dialog: FileDialog,
    /// Is the About dialog shown?
    pub about_dialog_shown: bool,

    /// Is the color picker mode on?
    pub picker_mode: bool,
    /// Coordinates of a point picked by the color picker
    pub picked_point: Option<[f32; 2]>,

    /// How far is the image zoomed?
    pub zoom: f32,
    /// Center of the image in the zoom
    pub center: Pos2,
}

impl State {
    /// Update the app state
    pub fn update(&mut self, renderer: &mut Renderer) {
        if let Some(items) = self.import_dialog.take_picked_multiple() {
            for item in items {
                self.path_send
                    .send((item, false))
                    .expect("Video parsing thread quit");
                self.files_being_imported += 1;
            }
        } else if let Some(item) = self.import_dialog.take_picked() {
            self.encoding_dialog.shown = false;
            if self.encode_cdng {
                self.start_encoding_cdng(item);
            } else {
                self.start_encoding(item);
            }
        }

        if let Some(item) = self.open_dialog.take_picked() {
            if self.open_dialog.mode() == DialogMode::SaveFile {
                self.update_current_file();
                if let Err(e) = File::create(&item)
                    .map_err(Into::into)
                    .and_then(|f| ciborium::into_writer(&self.files, f))
                {
                    eprintln!("{e}");
                }
            } else {
                match File::open(&item)
                    .map_err(Into::into)
                    .and_then(ciborium::from_reader)
                {
                    Ok(files) => {
                        self.files = files;
                    }
                    Err(e) => {
                        eprintln!("{e}");
                    }
                }
                for (_, file) in self.files.iter_mut() {
                    if let Some(ri) = file.raw_info {
                        file.focus_pixels =
                            load_focus_pixels(self.fpm.clone(), file.camera, ri.width, ri.height)
                    }
                }
            }
        }

        while let Ok((videofile, filename, switch)) = self.vid_recv.try_recv() {
            self.files_being_imported -= 1;
            let videofile = match videofile {
                Ok(video) => video,
                Err(e) => {
                    eprintln!("Error parsing {:?}: {e}", filename);
                    continue;
                }
            };
            if videofile.frames.is_empty() {
                continue;
            }
            let switch_file = if switch {
                filename.clone()
            } else {
                Default::default()
            };
            self.files.insert(filename, videofile);
            if switch {
                self.change_file(renderer, switch_file);
            }
        }

        let frame_len = self.frame_start.elapsed().as_micros() as f32;
        self.frame_start = Instant::now();
        let vid_frame_len = self.vid_frame_start.elapsed().as_micros() as f32;
        let ideal_len = self.ideal_frame_len + self.frame_delta;

        if let Some(image_ids) = self.image_ids {
            if self.compute.is_signaled() {
                // It's better for maintaining the desired FPS to present the frame now than later
                if !self.paused
                    && (vid_frame_len - ideal_len).abs()
                        < (vid_frame_len + frame_len - ideal_len).abs()
                {
                    self.vid_frame_start = Instant::now();
                    self.frame_delta = ideal_len - vid_frame_len;
                    /*println!(
                        "{} {} {} {}",
                        (s.first_start.elapsed().as_millis() * 23_977 / 1_000_000) % s.frames_len as u128,
                        s.frame_number,
                        vid_frame_len,
                        frame_len,
                    );*/
                    self.frame_number = self.frame_number + 1;
                    if self.frame_number >= self.trim.end {
                        self.frame_number = self.trim.start;
                    }

                    let (upload_buffer, _) = self.recv.recv().unwrap();
                    self.compute.process(
                        Some(upload_buffer),
                        self.second_img as usize,
                        self.pc,
                        None,
                    );
                    self.second_img = !self.second_img;
                    self.current_img = Some(image_ids[self.second_img as usize]);
                } else if self.paused {
                    self.second_img = !self.second_img;
                    self.current_img = Some(image_ids[self.second_img as usize]);
                    self.compute.pause();
                }
            } else if !self.paused {
                let (upload_buffer, _) = self.recv.recv().unwrap();

                self.compute.process(
                    Some(upload_buffer),
                    !self.second_img as usize,
                    self.pc,
                    None,
                );
            }
        }

        if self.paused
            && !self.compute.is_computing()
            && (self.pc != self.prev_pc || self.recompute)
        {
            let upload_buffer = if self.rewound_frame {
                loop {
                    let (upload_buffer, i) = self.recv.recv().unwrap();
                    if i == self.frame_number {
                        self.rewound_frame = false;
                        break Some(upload_buffer);
                    }
                }
            } else {
                None
            };

            self.compute
                .process(upload_buffer, !self.second_img as usize, self.pc, None);
            self.recompute = false;
        }

        let current_frame = self.frame_number;

        // Set immediate UI in redraw here
        renderer.begin_frame();
        let ctx = renderer.context();
        self.open_dialog.update(&ctx);
        self.import_dialog.update(&ctx);

        gui::layout(self, &ctx);

        if self.picker_mode {
            if let Some(xy) = self.picked_point.take() {
                if let Some(rgb) = self.compute.sample(xy) {
                    self.color_params.illuminant =
                        Illuminant::custom(self.color_params.rgb_to_xy(rgb));
                    self.picker_mode = false
                }
            }
        }

        self.prev_pc = self.pc;
        self.color_params
            .update_push_constants(&self.prev_color_params, &mut self.pc);
        self.prev_color_params = self.color_params;

        let time = self.first_start.elapsed().as_millis() as f64 / 1000.0;
        self.undo_color_params.feed_state(time, &self.color_params);
        self.undo_pc.add_undo(&self.pc);

        if let Some(new_file) = self.changed_file.take() {
            self.update_current_file();
            for (_, video) in self.files.iter_mut() {
                video.selected = false;
            }
            self.change_file(renderer, new_file);
        }

        if current_frame != self.frame_number {
            while self.recv.try_recv().is_ok() {}
            self.cmd_send
                .send(VideoCommand::Rewind(self.frame_number))
                .unwrap();
            self.recompute = true;
            self.rewound_frame = true;
        }
    }

    /// Update the parameters of the current file
    pub fn update_current_file(&mut self) {
        if let Some(current_file) = self.files.get_mut(&self.filename) {
            current_file.pc = self.pc;
            current_file.color_params = self.color_params;
            current_file.undo_pc = self.undo_pc.clone();
            current_file.undo_color_params = self.undo_color_params.clone();
            current_file.current_frame = self.frame_number;
            current_file.trim = self.trim.clone();
        }
    }

    /// Change the currently loaded file
    pub fn change_file(&mut self, gui: &mut Renderer, new_file: String) {
        self.filename = new_file;
        let video = self.files.get_mut(&self.filename).unwrap();
        video.selected = true;

        self.color_params = video.color_params;
        self.undo_color_params = video.undo_color_params.clone();
        self.pc = video.pc;
        self.undo_pc = video.undo_pc.clone();

        self.frame_number = video.current_frame;
        self.recompute = true;
        self.frames_len = video.frames.len();
        self.trim = video.trim.clone();
        self.extent = [video.width as u32, video.height as u32];
        // Play timelapses sped up
        let fps = if video.fps >= 16.0 { video.fps } else { 24.0 };
        self.ideal_frame_len = 1_000_000.0 / fps;
        let (snd, rcv) = sync_channel(2);
        self.cmd_send
            .send(VideoCommand::ChangeFile(video.clone().into(), snd))
            .unwrap();
        self.cmd_send
            .send(VideoCommand::Rewind(self.frame_number))
            .unwrap();
        self.recv = rcv;

        self.compute = Compute::new(self.extent, video.spec, self.gpu_context.clone());
        for id in self.image_ids.iter().flatten() {
            gui.unregister_image(*id);
        }
        self.image_ids = Some(array::from_fn(|i| {
            gui.register_image(
                self.compute.out_image(i),
                SamplerCreateInfo {
                    mag_filter: Filter::Linear,
                    min_filter: Filter::Linear,
                    ..Default::default()
                },
            )
        }));
        self.current_img = None;

        // Stretch the extent, so that the video displays correctly
        self.extent[0] *= video.column_binning as u32;
    }

    /// Spawn the thread sending the processed frames to ffmpeg
    fn start_encoding(&mut self, path: PathBuf) {
        self.update_current_file();

        let to_encode = self
            .files
            .iter()
            .filter(|(_, m)| m.selected)
            .map(|(n, m)| (n.clone(), m.clone()))
            .collect();
        let cloned_context = self.gpu_context.clone();

        let video_args = if let Some(i) = self.encoding_dialog.current_video_recipe {
            let mut v = self.encoding_dialog.video_recipes[i].args.clone();
            v.extend(
                self.encoding_dialog.video_recipes[i]
                    .params
                    .iter()
                    .flat_map(|p| p.to_args().into_iter()),
            );
            v
        } else {
            Default::default()
        };

        let audio_args = if let Some(i) = self.encoding_dialog.current_audio_recipe {
            let mut a = self.encoding_dialog.audio_recipes[i].args.clone();
            a.extend(
                self.encoding_dialog.audio_recipes[i]
                    .params
                    .iter()
                    .flat_map(|p| p.to_args().into_iter()),
            );
            a
        } else {
            Default::default()
        };

        let es = self.encoding_state.clone();
        let resize =
            Some(self.encoding_dialog.resize).filter(|_| self.encoding_dialog.allow_resize);
        let fps = Some(self.encoding_dialog.fps).filter(|_| self.encoding_dialog.allow_fps);
        thread::spawn(move || {
            env::set_current_dir(path).expect("Cannot change the directory for export");
            encoding::encode(
                to_encode,
                cloned_context,
                es,
                (video_args, audio_args),
                resize,
                fps,
            )
        });
    }

    /// Spawn the thread sending the processed frames to be encoded as CinemaDNG
    fn start_encoding_cdng(&mut self, path: PathBuf) {
        self.update_current_file();

        let to_encode = self
            .files
            .iter()
            .filter(|(_, m)| m.selected)
            .map(|(n, m)| (n.clone(), m.clone()))
            .collect();
        let es = self.encoding_state.clone();
        thread::spawn(move || {
            env::set_current_dir(path).expect("Cannot change the directory for export");
            encoding::encode_cdngs(to_encode, es)
        });
    }

    /// Undo the current color correction
    pub fn undo(&mut self) {
        if let Some(cp) = self.undo_color_params.undo(&self.color_params) {
            self.color_params = *cp
        }
        if let Some(pc) = self.undo_pc.undo(&self.pc) {
            self.pc = *pc
        }
        self.recompute = true;
    }

    /// Redo the current color correction
    pub fn redo(&mut self) {
        if let Some(cp) = self.undo_color_params.redo(&self.color_params) {
            self.color_params = *cp
        }
        if let Some(pc) = self.undo_pc.redo(&self.pc) {
            self.pc = *pc
        }
        self.recompute = true;
    }

    /// Do we have an undo point different than the current state?
    pub fn has_undo(&mut self) -> bool {
        self.undo_color_params.has_undo(&self.color_params) || self.undo_pc.has_undo(&self.pc)
    }

    /// Do we have an undo point different than the current state?
    pub fn has_redo(&mut self) -> bool {
        self.undo_color_params.has_redo(&self.color_params) || self.undo_pc.has_redo(&self.pc)
    }

    /// Is there a need to do a redraw every frame?
    pub fn redraw(&self) -> bool {
        !self.paused
            || self.recompute
            || self.undo_pc.has_undo(&self.pc)
            || self.encoding_state.lock().unwrap().running
            || self.files_being_imported > 0
    }

    /// Set the frame as the first frame of the trimmed video
    pub fn mark_in(&mut self, frame: usize) {
        self.trim.start = frame;
        if frame > self.trim.end {
            self.trim.end = self.frames_len;
        }
        let _ = self.cmd_send.send(VideoCommand::Trim(self.trim.clone()));
    }

    /// Set the frame as the last frame of the trimmed video
    pub fn mark_out(&mut self, frame: usize) {
        let frame = frame + 1;
        self.trim.end = frame;
        if frame < self.trim.start {
            self.trim.start = 0;
        }
        let _ = self.cmd_send.send(VideoCommand::Trim(self.trim.clone()));
    }
}
