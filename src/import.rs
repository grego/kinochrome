use dng::DngReaderError;
use dng::ifd::{Ifd, IfdValue};
use dng::{DngReader, tags::ifd};
use egui::util::undoer::Undoer;
use ljpeg::{Decoder, DecoderError};
use mlv::{
    FileHeader,
    frame::{Frame, Header, RawInfo, WavInfo},
};
use serde::{Deserialize, Serialize};
use vulkano::buffer::Subbuffer;
use vulkano::memory::allocator::MemoryAllocator;

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, BufReader, ErrorKind, Read, Seek, SeekFrom};
use std::mem::swap;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::{
    Arc, Mutex,
    mpsc::{Receiver, Sender, SyncSender},
};

use crate::color_utils::{ColorParams, Illuminant, array_to_mat3, identity_mat, inv3, transpose};
use crate::gpu_compute::{PushConstantData, Specialization, make_upload_buffer};

/// Video file
#[derive(Clone, Deserialize, Serialize)]
pub struct VideoFile {
    /// Filesystem path of the video
    pub path: Arc<Path>,
    /// Camera model ID, 0 if unavailable
    pub camera: u32,
    /// Video frames
    pub frames: Frames,
    /// Audio frames
    pub audio_frames: Arc<[AudioFrame]>,
    /// WAV audio info
    pub wav_info: Option<WavInfo>,
    /// MLV raw info
    pub raw_info: Option<RawInfo>,
    /// Delay of the audio with respect to the video
    pub audio_delay: i64,
    /// Compression used
    pub compression: Compression,
    /// The number of bits per each pixel
    pub bits_per_pixel: u8,
    /// An array of focus pixel coordinates for correction
    #[serde(skip)]
    pub focus_pixels: Arc<[[u16; 2]]>,
    /// Video width
    pub width: usize,
    /// Video height
    pub height: usize,
    /// Number of frames per second
    pub fps: f32,
    /// How many columns of pixels are binned together in each row
    pub column_binning: u8,
    /// Specialization constants for compute shader
    pub spec: Specialization,
    /// The current push constants
    pub pc: PushConstantData,
    /// The current color parameters
    pub color_params: ColorParams,
    /// Parameters for color grading
    pub undo_color_params: Undoer<ColorParams>,
    /// Push constants for compute shader
    pub undo_pc: Undoer<PushConstantData>,
    /// The number of the current frame
    pub current_frame: usize,
    /// Is the video selected?
    pub selected: bool,
    /// The first and the last frame the video should be trimmed to.
    pub trim: Range<usize>,
}

/// A map of focus pixels, indexed by camera model, width and height
pub type FocusPixelMap = Arc<Mutex<HashMap<(u32, u32, u32), Arc<[[u16; 2]]>>>>;

/// Video frames
#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum Frames<T = ()> {
    /// MLV frames
    /// Might optionally contain the opened video file
    Mlv(Arc<[MlvVideoFrame]>, T),
    /// CinemaDNG frames - a list of files
    Dng(Arc<[PathBuf]>),
}

/// Single MLV frame
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct MlvVideoFrame {
    /// Position in the file
    pub pos: u64,
    /// Lenght of the frame
    pub len: usize,
    /// Video pan in the frame
    pub pan: [u16; 2],
}

/// MLV audio frame
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct AudioFrame {
    /// Position in the file
    pub pos: u64,
    /// Lenght
    pub len: usize,
}

/// Command for the video decoder thread
pub enum VideoCommand {
    /// Change the video file
    ChangeFile(Box<VideoFile>, SyncSender<(Subbuffer<[u16]>, usize)>),
    /// Rewind to the provided frame
    Rewind(usize),
    /// Only decode the frames in the specified interval
    Trim(Range<usize>),
}

/// Compression scheme
#[derive(Clone, Copy, Deserialize, Serialize)]
pub enum Compression {
    /// No compression - only bit packing.
    /// True if big endian, false if little endian.
    None(bool),
    /// Lossles JPEG
    LJpeg,
}

/// A loop receiving video files and parsing them
pub fn parse_videos(
    recv: Receiver<(PathBuf, bool)>,
    send: Sender<(io::Result<VideoFile>, String, bool)>,
    fpm: FocusPixelMap,
) {
    while let Ok((path, switch)) = recv.recv() {
        let filename = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into();
        if let Err(e) = send.send((parse_video(&path, fpm.clone()), filename, switch)) {
            eprint!("{e}");
            return;
        }
    }
}

/// Parse a video file
pub fn parse_video(path: &Path, fpm: FocusPixelMap) -> Result<VideoFile, io::Error> {
    if path.is_dir() {
        parse_cdng(path)
    } else {
        parse_mlv(path, fpm)
    }
}

/// Parse a MLV file
pub fn parse_mlv(path: &Path, fpm: FocusPixelMap) -> Result<VideoFile, io::Error> {
    let path = path.canonicalize()?;
    let vidfile = File::open(&path)?;
    let filelen = vidfile.metadata()?.len();
    let mut vidfile = BufReader::new(vidfile);
    let (header, _) = FileHeader::read(&mut vidfile)?;
    let compression = match header.video_class {
        1 => Compression::None(false),
        33 => Compression::LJpeg,
        _ => {
            return Err(io::Error::new(
                ErrorKind::Unsupported,
                "unsuported compression",
            ));
        }
    };
    dbg!(header);

    let mut black_level = 0.0;
    let mut white_level = 1.0;
    let mut cam_matrix = identity_mat();
    let mut temperature = 6503;
    let frame_count = header.video_frame_count as usize;
    let mut frames = vec![
        MlvVideoFrame {
            pos: 0,
            len: 0,
            pan: [0, 0]
        };
        header.video_frame_count as usize
    ];
    let mut audio_frames = Vec::with_capacity(header.audio_frame_count as usize);
    let mut wav_info = None;
    let mut raw_info = None;
    let (mut width, mut height) = (0, 0);
    let (mut raw_w, mut raw_h) = (0, 0);
    let mut column_binning = 1;
    let mut camera = 0;
    let mut bits_per_pixel = 14;
    let fps: f32 = header.fps.into();

    let mut video_start = 0;
    let mut audio_start = 0;

    loop {
        let (frame, timestamp) = match Frame::read_position(&mut vidfile) {
            Ok(fc) => fc,
            Err(e) if e.kind() == ErrorKind::UnexpectedEof => {
                vidfile.rewind()?;
                break;
            }
            Err(e) => panic!("{}", e),
        };

        match frame.header {
            Header::RawInfo(ri) => {
                black_level = ri.black_level as f32 / 65535.0;
                white_level = ri.white_level as f32 / 65535.0;
                cam_matrix = transpose(inv3(array_to_mat3(ri.color_matrix.map(Into::into))));
                dbg!(cam_matrix);
                width = ri.res_x as usize;
                height = ri.res_y as usize;
                raw_w = ri.width;
                raw_h = ri.height;
                bits_per_pixel = ri.bits_per_pixel as u8;

                raw_info = Some(ri);

                dbg!(ri);
            }
            Header::WhiteBalance(wb) => {
                temperature = wb.kelvin as u16;
                dbg!(wb);
            }
            Header::Idnt(id) => {
                camera = id.camera_model;
            }
            Header::RawCapture(rc) => {
                column_binning = rc.binning_x / (1 + rc.skipping_y);
                dbg!(rc);
            }
            Header::Exposure(exp) => {
                dbg!(exp);
            }
            Header::DualIso(diso) => {
                dbg!(diso);
            }
            Header::WavInfo(wi) => {
                wav_info = Some(wi);
            }
            Header::Video(v) => {
                if video_start == 0 {
                    video_start = timestamp;
                }
                let i = v.number as usize;
                if i >= frame_count {
                    continue;
                }
                let pan = [v.pan_x, v.pan_y];
                let pos = frame.payload.start();
                let len = frame.payload.length();
                // Seek doesn't check whether the EOF is reached
                if pos + frame.payload.full_length as u64 > filelen {
                    break;
                }
                frames[i] = MlvVideoFrame { pos, len, pan };
            }
            Header::Audio(_) => {
                if audio_start == 0 {
                    audio_start = timestamp;
                }
                let pos = frame.payload.start();
                let len = frame.payload.length();
                // Seek doesn't check whether the EOF is reached
                if pos + frame.payload.full_length as u64 > filelen {
                    break;
                }
                audio_frames.push(AudioFrame { pos, len });
            }
            _ => {}
        }

        vidfile.seek(SeekFrom::Current(frame.payload.full_length as i64))?;
    }

    for i in 1..frame_count {
        if frames[i].len == 0 {
            frames[i] = frames[i - 1];
        }
    }

    let stretch = 1.0 / (white_level - black_level);
    let spec = Specialization {
        first_red: [0, 0],
        black_level,
        stretch,
    };
    let color_params = ColorParams {
        cam_matrix,
        illuminant: Illuminant::D(temperature),
        ..Default::default()
    };
    let pc = color_params.set_push_constants(&Default::default());

    let mut undo_color_params: Undoer<_> = Default::default();
    undo_color_params.add_undo(&color_params);
    let mut undo_pc: Undoer<_> = Default::default();
    undo_pc.add_undo(&pc);

    let focus_pixels = load_focus_pixels(fpm, camera, raw_w, raw_h);

    let len = frames.len();
    Ok(VideoFile {
        path: path.into(),
        camera,
        frames: Frames::Mlv(frames.into(), ()),
        audio_frames: audio_frames.into(),
        wav_info,
        raw_info,
        audio_delay: video_start as i64 - audio_start as i64,
        compression,
        bits_per_pixel,
        focus_pixels,
        width,
        height,
        fps,
        column_binning,
        spec,
        undo_color_params,
        undo_pc,
        pc,
        color_params,
        current_frame: 0,
        trim: 0..len,
        selected: false,
    })
}

/// Parse a CinemaDNG directory
pub fn parse_cdng(path: &Path) -> Result<VideoFile, io::Error> {
    let mut entries: Vec<_> = fs::read_dir(path)?
        .filter_map(|entry| {
            let e = entry.ok()?;
            let path = e.path();
            if e.file_type().ok()?.is_file() && path.extension()?.eq_ignore_ascii_case("dng") {
                Some(path)
            } else {
                None
            }
        })
        .collect();
    entries.sort_unstable();
    if entries.is_empty() {
        return Err(io::Error::new(ErrorKind::NotFound, "no DNG file found"));
    }

    let dng = match DngReader::read(File::open(&entries[0])?) {
        Ok(dng) => dng,
        Err(DngReaderError::IoError(e)) => return Err(e),
        Err(DngReaderError::FormatError(s)) => return Err(io::Error::other(s)),
        Err(DngReaderError::Other(s)) => return Err(io::Error::other(s)),
    };
    let ifd = dng.get_ifd0();
    let mut first_red = [0, 0];
    let mut black_level = 0.0;
    let mut white_level = 1.0;
    let mut cam_matrix = identity_mat::<3>();
    let (mut width, mut height) = (0, 0);
    let mut fps = 24.0;
    let mut bits_per_pixel = 14;
    let mut compression = Compression::None(true);
    let err = || io::Error::other("incorrect IFD tag value type");
    let mut read_ifd = |ifd: &Ifd| {
        for entry in ifd.entries() {
            let tag = entry.tag.numeric();
            if tag == ifd::ImageWidth.tag {
                width = entry.value.as_u32().ok_or_else(err)? as usize;
            } else if tag == ifd::ImageLength.tag {
                height = entry.value.as_u32().ok_or_else(err)? as usize;
            } else if tag == ifd::FrameRate.tag {
                fps = entry.value.as_f64().ok_or_else(err)? as f32;
            } else if tag == ifd::BitsPerSample.tag {
                if let Some(bpp) = entry.value.as_u32() {
                    bits_per_pixel = bpp as u8;
                }
            } else if tag == ifd::Compression.tag {
                match entry.value.as_u32().ok_or_else(err)? {
                    1 => {
                        compression = Compression::None(true);
                    }
                    7 => {
                        compression = Compression::LJpeg;
                    }
                    x => return Err(io::Error::other(format!("Compression {x} not implemented"))),
                }
            } else if tag == ifd::CFAPattern.tag {
                for (i, val) in entry.value.as_list().enumerate() {
                    if val.as_u32().ok_or_else(err)? == 0 {
                        first_red = [(i % 2) as u16, (i / 2) as u16];
                        break;
                    }
                }
            } else if tag == ifd::BlackLevel.tag {
                black_level = entry.value.as_u32().ok_or_else(err)? as f32 / ((1 << 16) - 1) as f32;
            } else if tag == ifd::WhiteLevel.tag {
                white_level = entry.value.as_u32().ok_or_else(err)? as f32 / ((1 << 16) - 1) as f32;
            } else if tag == ifd::ColorMatrix1.tag {
                let cm: Vec<_> = entry
                    .value
                    .as_list()
                    .filter_map(|v| v.as_f64())
                    .map(|v| v as f32)
                    .collect();
                if let Ok(mat) = cm.try_into() {
                    cam_matrix = transpose(inv3(array_to_mat3(mat)));
                } else {
                    eprintln!("Warning: camera matrix has invalid size; skipping")
                }
            }
        }
        Ok(())
    };
    read_ifd(ifd)?;
    if let Some(IfdValue::Ifd(ifd)) = ifd
        .get_entry_by_path(&dng.main_image_data_ifd_path())
        .map(|e| e.value)
    {
        read_ifd(ifd)?;
    }

    if white_level == 1.0 {
        white_level = ((1 << bits_per_pixel) as f32) / ((1 << 16) as f32);
    }
    let stretch = 1.0 / (white_level - black_level);
    let spec = Specialization {
        first_red,
        black_level,
        stretch,
    };
    let color_params = ColorParams {
        cam_matrix,
        ..Default::default()
    };
    let pc = color_params.set_push_constants(&Default::default());

    let mut undo_color_params: Undoer<_> = Default::default();
    undo_color_params.add_undo(&color_params);
    let mut undo_pc: Undoer<_> = Default::default();
    undo_pc.add_undo(&pc);
    let len = entries.len();
    Ok(VideoFile {
        path: path.into(),
        camera: 0,
        frames: Frames::Dng(entries.into()),
        audio_frames: Default::default(),
        wav_info: None,
        raw_info: None,
        audio_delay: 0,
        bits_per_pixel,
        compression,
        focus_pixels: Default::default(),
        width,
        height,
        fps,
        column_binning: 1,
        spec,
        undo_color_params,
        undo_pc,
        pc,
        color_params,
        current_frame: 0,
        selected: false,
        trim: 0..len,
    })
}

/// Decode the image into the `out` buffer using the provided compression scheme
pub fn decode_image(
    payload: &[u8],
    output: &mut [u16],
    (compression, bits_per_pixel): (Compression, u8),
    [width, height]: [usize; 2],
    fp: &[[u16; 2]],
    pan: [u16; 2],
) -> Result<(), DecoderError> {
    match compression {
        Compression::None(true) => unpack_bits::<true>(payload, bits_per_pixel, output),
        Compression::None(false) => unpack_bits::<false>(payload, bits_per_pixel, output),
        Compression::LJpeg => {
            let decoder = Decoder::new(payload)?;
            if width * height != decoder.width() * decoder.height() * decoder.components() {
                eprintln!(
                    "Expected dimensions {width}x{height}, decoded {}x{}",
                    decoder.width(),
                    decoder.height()
                );
                return Err(DecoderError::SmallBuffer);
            }

            //let start = std::time::Instant::now();
            decoder.decode_to_buffer(output, decoder.width(), decoder.height())?;
            //println!("Done in {}ms.", start.elapsed().as_micros() as f64 / 1000.0);
        }
    }

    let crop_x = (pan[0] + 7) & !7;
    let crop_y = pan[1] & !1;
    for &[x, y] in fp {
        let x = x.saturating_sub(crop_x);
        let y = y.saturating_sub(crop_y);
        interpolate_pixel(output, x as usize, y as usize, width, height);
    }

    Ok(())
}

/// A function to run in a decoder thread to read the video frames
pub fn read_frames(
    allocator: Arc<dyn MemoryAllocator>,
    cmd_recv: Receiver<VideoCommand>,
    looped: bool,
) {
    let (mut video, mut send) = loop {
        if let VideoCommand::ChangeFile(m, s) = cmd_recv.recv().unwrap() {
            break (m, s);
        }
    };
    let mut vidframes = match video.frames {
        Frames::Mlv(ref frames, _) => Frames::Mlv(frames.clone(), File::open(&video.path).unwrap()),
        Frames::Dng(ref frames) => Frames::Dng(frames.clone()),
    };
    let mut trim = video.trim;
    let mut i = 0;
    loop {
        while let Ok(cmd) = cmd_recv.try_recv() {
            match cmd {
                VideoCommand::Rewind(j) => {
                    i = j % video.frames.len();
                }
                VideoCommand::ChangeFile(m, s) => {
                    video = m;
                    send = s;
                    vidframes = match video.frames {
                        Frames::Mlv(ref frames, _) => {
                            Frames::Mlv(frames.clone(), File::open(&video.path).unwrap())
                        }
                        Frames::Dng(ref frames) => Frames::Dng(frames.clone()),
                    };
                }
                VideoCommand::Trim(t) => {
                    trim = t;
                }
            }
        }

        let (payload, pan) = vidframes.read_ith(i);
        let upload_buffer = make_upload_buffer(video.width, video.height, allocator.clone());
        let mut output = upload_buffer.write().unwrap();
        let extent = [video.width, video.height];
        if let Err(e) = decode_image(
            &payload,
            &mut output,
            (video.compression, video.bits_per_pixel),
            extent,
            &video.focus_pixels,
            pan,
        ) {
            eprint!("{e}");
        };
        drop(output);

        // The receiver may be dropped if the file was changed meanwhile.
        let _ = send.send((upload_buffer, i));

        i += 1;
        if i >= trim.end {
            if looped {
                i = trim.start;
            } else {
                return;
            }
        }
    }
}

impl<T> Frames<T> {
    /// Number of the frames
    pub fn len(&self) -> usize {
        match self {
            Frames::Mlv(f, _) => f.len(),
            Frames::Dng(f) => f.len(),
        }
    }

    /// Are there any frames
    pub fn is_empty(&self) -> bool {
        match self {
            Frames::Mlv(f, _) => f.is_empty(),
            Frames::Dng(f) => f.is_empty(),
        }
    }
}

impl Frames<File> {
    /// Return the i-th frame and the panning coordinates, if available.
    pub fn read_ith(&mut self, i: usize) -> (Vec<u8>, [u16; 2]) {
        match *self {
            Frames::Mlv(ref frames, ref mut vidfile) => {
                let MlvVideoFrame { pos, len, pan } = frames[i];
                vidfile.seek(SeekFrom::Start(pos)).unwrap();
                let mut payload = vec![0; len];
                vidfile.read_exact(&mut payload).unwrap();
                (payload, pan)
            }
            Frames::Dng(ref frames) => {
                let dng = DngReader::read(File::open(&frames[i]).unwrap()).unwrap();
                let main_ifd = dng.main_image_data_ifd_path();
                let len = dng.needed_buffer_length_for_image_data(&main_ifd).unwrap();
                let mut payload = vec![0u8; len];
                dng.read_image_data_to_buffer(&main_ifd, &mut payload)
                    .unwrap();
                (payload, [0, 0])
            }
        }
    }
}

#[allow(dead_code)]
/// Interpolate using the method from rewind - there is a bug
pub fn interpolate_rewind(image_data: &mut [u16], x: usize, y: usize, w: usize, h: usize) {
    if (x < 3) || (x > (w - 4)) || (y < 3) || (y > (h - 4)) {
        return;
    }

    // 1. Retrieve vectors from 7x7 kernel
    // d[0] — vertical vector
    // d[1] — horizontal vector
    // index reference:
    //        paper     -3 -2 -1 0 +1 +2 +3
    //        actual     0  1  2    3  4  5
    let mut d: [[u16; 6]; 2] = [
        [
            image_data[x + ((y - 3) * w)],
            image_data[x + ((y - 2) * w)],
            image_data[x + ((y - 1) * w)],
            image_data[x + ((y + 1) * w)],
            image_data[x + ((y + 2) * w)],
            image_data[x + ((y + 3) * w)],
        ],
        [
            image_data[x - 3 + (y * w)],
            image_data[x - 2 + (y * w)],
            image_data[x - 1 + (y * w)],
            image_data[x + 1 + (y * w)],
            image_data[x + 2 + (y * w)],
            image_data[x + 3 + (y * w)],
        ],
    ];

    // 2,3 — We don't need these stepse because of diagonal af dots arrangement

    // 4. Normalizing vectors
    // vertical norm.
    d[0][2] = d[0][1] + ((d[0][2] - d[0][0]) / 2);
    d[0][3] = d[0][4] + ((d[0][3] - d[0][5]) / 2);
    // horizontal norm.
    d[1][2] = d[1][1] + ((d[1][2] - d[1][0]) / 2);
    d[1][3] = d[1][4] + ((d[1][3] - d[1][5]) / 2);

    // 5. Deltas and Weights
    let d_vert = d[0][2].abs_diff(d[0][3]) as f32;
    let d_horiz = d[1][2].abs_diff(d[1][3]) as f32;
    let delta = d_vert + d_horiz;

    let mut w_vert = f32::clamp(1.0 - (d_vert / delta), 0.0, 1.0);
    let mut w_horiz = f32::clamp(1.0 - (d_horiz / delta), 0.0, 1.0);
    let too_much = (w_vert + w_horiz - 1.0) / 2.0;
    if too_much > 0.0 {
        w_vert -= too_much;
        w_horiz -= too_much;
    }

    // 7. Calculating new pixel value
    let new_val =
        w_vert * ((d[0][2] + d[0][3]) as f32 / 2.0) + w_horiz * ((d[1][2] + d[1][3]) as f32 / 2.0);
    image_data[x + (y * w)] = (new_val as i32).clamp(0, 65_535) as u16;
}

/* find color of the raw pixel */
fn find_color(row: isize, col: isize) -> u8 {
    if (row % 2) == 0 && (col % 2) == 0 {
        0 /* red */
    } else if (row % 2) == 1 && (col % 2) == 1 {
        2 /* blue */
    } else {
        1 /* green */
    }
}

/// Inspired by the method from raw2dng, which computes the median of the neighbours.
/// This computes the average of neighbours, summed in the groups of 4, without the extreme
/// (minimum and maximum) group.
pub fn interpolate_pixel(image_data: &mut [u16], x: usize, y: usize, w: usize, h: usize) {
    if (x < 4) || (x >= (w - 4)) || (y < 4) || (y >= (h - 4)) {
        return;
    }

    //let mut neighbours = [0; 80];
    let mut k = 0;
    let mut sum: u32 = 0;
    //let (mut min, mut max) = (u32::MAX, 0);
    let (x, y, w, _h) = (x as isize, y as isize, w as isize, h as isize);
    let fc0 = find_color(x, y);

    let mut pick_middle = |mut a, mut b, mut c, mut d| {
        if b < a {
            swap(&mut a, &mut b);
        }
        if d < c {
            swap(&mut c, &mut d);
        }

        if a < c {
            swap(&mut a, &mut c);
        }
        if b > d {
            swap(&mut b, &mut d);
        }

        sum += a as u32 + b as u32;
        k += 2;
    };

    for i in 1..=4 {
        for j in 1..=4 {
            // examine only the neighbours of the same color
            if find_color(x + j, y + i) != fc0 {
                continue;
            }

            //neighbours[k] = image_data[(x + j + (y + i) * w) as usize];
            let a = image_data[(x - j + (y - i) * w) as usize];
            let b = image_data[(x + j + (y - i) * w) as usize];
            let c = image_data[(x - j + (y + i) * w) as usize];
            let d = image_data[(x + j + (y + i) * w) as usize];

            pick_middle(a, b, c, d);
        }
    }

    for j in 1..=4 {
        // examine only the neighbours of the same color
        if find_color(x + j, y) != fc0 {
            continue;
        }

        //neighbours[k] = image_data[(x + j + (y + i) * w) as usize];
        let a = image_data[(x - j + y * w) as usize];
        let b = image_data[(x + j + y * w) as usize];
        let c = image_data[(x + (y - j) * w) as usize];
        let d = image_data[(x + (y + j) * w) as usize];

        pick_middle(a, b, c, d);
    }

    //(&mut neighbours[..k]).sort_unstable();
    //sum -= max[0] + max[1] + min[0] + min[1];
    //if u32::abs_diff((sum / k), ((sum - min - max) / (k - 8))) > 10 {
    //    dbg!((sum, max, min, k, sum / k, (sum - max -min) / (k -8)));
    //}
    image_data[(x + y * w) as usize] = (sum / k) as u16;
}

/// Load focus pixel based on the camera id, raw width and height
pub fn load_focus_pixels(map: FocusPixelMap, id: u32, w: u32, h: u32) -> Arc<[[u16; 2]]> {
    if id == 0 {
        return [].into();
    }

    map.lock()
        .unwrap()
        .entry((id, w, h))
        .or_insert_with(|| {
            let fpmfile = format!("pixel_maps/{:x}_{}x{}.fpm", id, w, h);
            let focus_pixels = match read_fpm(&fpmfile) {
                Ok(fp) => fp,
                Err(e) => {
                    eprintln!("Error reading {}: {}", fpmfile, e);
                    Vec::with_capacity(0)
                }
            };
            focus_pixels.into()
        })
        .clone()
}

/// Read a list of focus pixel coordinates
fn read_fpm(name: &str) -> Result<Vec<[u16; 2]>, io::Error> {
    let s = fs::read_to_string(name)?;
    let mut iter = s.split_whitespace().map(|n| n.parse::<u16>().ok());
    let mut pixels = Vec::with_capacity(iter.size_hint().0 / 2);
    loop {
        let p = iter
            .next()
            .flatten()
            .and_then(|x| Some([x, iter.next().flatten()?]));
        match p {
            Some(p) => pixels.push(p),
            None => break,
        }
    }
    Ok(pixels)
}

#[allow(dead_code)]
fn show(bs: &[u8]) -> String {
    let mut visible = String::new();
    for &b in bs {
        let part: Vec<u8> = std::ascii::escape_default(b).collect();
        visible.push_str(std::str::from_utf8(&part).unwrap());
    }
    visible
}

fn unpack_bits<const BE: bool>(buffer: &[u8], bits: u8, out: &mut [u16]) {
    let (mut rem, mut rem_bits) = (0, 0);
    let mut i = 0;
    for c in buffer.chunks_exact(2).take(out.len() * bits as usize / 16) {
        let w = if BE {
            u16::from_be_bytes([c[0], c[1]])
        } else {
            u16::from_le_bytes([c[0], c[1]])
        };
        out[i] = (rem_bits << (bits - rem)) | (w >> (16 - (bits - rem)));
        rem = 16 - (bits - rem);
        rem_bits = w & ((1 << rem) - 1);
        i += 1;

        if rem >= bits {
            out[i] = rem_bits >> (rem - bits);
            rem -= bits;
            rem_bits = w & ((1 << rem) - 1);
            i += 1;
        }
    }
    if i < out.len() {
        out[i] = rem_bits << (bits - rem);
    }
}
