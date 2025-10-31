use crate::gpu_compute::{Compute, GpuContext, make_output_buffer};
use crate::import::{AudioFrame, Frames, VideoCommand, VideoFile, decode_image, read_frames};
use crate::state::log_error;

use std::array;
use std::fs::{self, File, read_to_string};
use std::io::{BufWriter, Error, ErrorKind, Read, Result, Seek, SeekFrom, Write};
use std::panic;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::mpsc::{Receiver, channel, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread;

use dng::ifd::{Ifd, IfdValue};
use dng::tags::{IfdType, ifd};
use dng::{DngWriter, FileType};
use ljpeg::{Bitdepth, Components, Encoder, Predictor};
use mlv::Fraction;
use mlv::frame::RawInfo;
use serde::{Deserialize, Deserializer};

use packbytes::ToBytes;

/// State of the video encoder
#[derive(Clone, Debug, Default)]
pub struct EncodingState {
    /// Total number of files in this render.
    pub num_files: usize,
    /// The number of the current file being rendered.
    pub cur_file: usize,
    /// Total number of frames
    pub num_frames: usize,
    /// The number of the current frame
    pub cur_frame: usize,
    /// Is the encoding running?
    pub running: bool,
    /// Is the encoding to be aborted?
    pub abort: bool,
}

/// Recipe for ffmpeg encoding
#[derive(Clone, Debug, Deserialize)]
pub struct Recipe {
    /// Name of the recipe visible to the user
    pub name: String,
    /// Kind of the recipe
    pub kind: RecipeKind,
    /// Arguments to ffmpeg
    #[serde(deserialize_with = "deserialize_shell_args", default)]
    pub args: Vec<String>,
    /// Adjustable parameters for the ffmpeg arguments
    #[serde(default)]
    pub params: Vec<Param>,
}

/// What is the recipe for?
#[derive(Clone, Debug, Deserialize)]
pub enum RecipeKind {
    /// A video codec
    Video,
    /// An audio codec
    Audio,
}

/// A parameter for an encoding recipe
#[derive(Clone, Debug, Deserialize)]
pub struct Param {
    /// Name of the parameter, displayed to the user
    pub name: String,
    /// The argument before the parameter
    pub arg: String,
    /// Value of the parameter
    #[serde(flatten)]
    pub value: ParamValue,
}

/// Value of the parameter
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type")]
pub enum ParamValue {
    /// Integer
    Int {
        /// The minimal allowed value
        min: i64,
        /// The maximal allowed value
        max: i64,
        /// The current value, called `default` in the recipe
        #[serde(rename = "default")]
        value: i64,
    },
    /// Floating point
    Float {
        /// The minimal allowed value
        min: f64,
        /// The maximal allowed value
        max: f64,
        /// The current value, called `default` in the recipe
        #[serde(rename = "default")]
        value: f64,
    },
    /// Set of values
    Discrete {
        /// Possible options: `[string, description]`
        options: Arc<[[String; 2]]>,
        /// The index of the default option, called `default` in the recipe
        #[serde(rename = "default")]
        index: usize,
    },
}

#[derive(Default)]
/// A window dialog for ffmpeg video encoding
pub struct EncodingDialog {
    /// A list of available video recipes
    pub video_recipes: Vec<Recipe>,
    /// The index of the currently selected video recipe
    pub current_video_recipe: Option<usize>,
    /// A list of available audio recipes
    pub audio_recipes: Vec<Recipe>,
    /// The index of the currently selected audio recipe
    pub current_audio_recipe: Option<usize>,
    /// Is the dialog shown?
    pub shown: bool,
    /// Allow resizing the videos
    pub allow_resize: bool,
    /// Link width and height when resizing
    pub link_wh: bool,
    /// How to resize the videos
    pub resize: [u32; 2],
    /// Allow changing the video fps
    pub allow_fps: bool,
    /// Video FPS to change
    pub fps: f32,
}

/// Encode videos using ffmpeg
pub fn encode(
    files: Vec<(String, VideoFile)>,
    gpu_context: GpuContext,
    state: Arc<Mutex<EncodingState>>,
    (video_args, audio_args): (Vec<String>, Vec<String>),
    resize: Option<[u32; 2]>,
    fps: Option<f32>,
) {
    {
        let mut s = state.lock().unwrap();
        s.running = true;
        s.num_files = files.len();
        s.cur_file = 0;
    }
    for (name, video) in files {
        {
            let mut s = state.lock().unwrap();
            s.cur_file += 1;
            if s.abort {
                s.abort = false;
                break;
            }
        }
        if let Err(e) = encode_file(
            &name,
            video,
            gpu_context.clone(),
            &state,
            (&video_args, &audio_args),
            resize,
            fps,
        ) {
            log_error(&e, &format!("exporting {name}"));
        }
    }
    state.lock().unwrap().running = false;
}

fn encode_file(
    name: &str,
    video: VideoFile,
    gpu_context: GpuContext,
    state: &Mutex<EncodingState>,
    (video_args, audio_args): (&[String], &[String]),
    mut resize: Option<[u32; 2]>,
    fps: Option<f32>,
) -> Result<()> {
    let cloned_alloc = gpu_context.memory_alloc.clone();
    let (width, height, fps) = (video.width, video.height, fps.unwrap_or(video.fps));
    let pc = video.pc;
    let spec = video.spec;
    let column_binning = video.column_binning;
    let frames = video.trim.clone();

    let wav_name = format!("{}.wav", name);
    let has_audio = create_wav(&wav_name, &video)?;

    let (cmd_send, cmd_recv) = channel();
    thread::spawn(move || read_frames(cloned_alloc, cmd_recv, false));
    let (send, recv) = sync_channel(3);
    cmd_send
        .send(VideoCommand::ChangeFile(video.into(), send))
        .unwrap();
    cmd_send.send(VideoCommand::Rewind(frames.start)).unwrap();

    if resize.is_none() && column_binning > 1 {
        resize = Some([width as u32 * column_binning as u32, height as u32]);
    }
    let [w, h] = resize.unwrap_or([width as u32, height as u32]);
    let output_buffers: [_; 2] = array::from_fn(|_| {
        make_output_buffer(w as usize, h as usize, gpu_context.memory_alloc.clone())
    });

    let mut compute = Compute::new([width as u32, height as u32], spec, gpu_context);

    let file_stem = Path::new(&name)
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy();
    let mut cmd = Command::new("ffmpeg");
    cmd.stdin(Stdio::piped());
    cmd.args(["-f", "rawvideo"])
        .args(["-pix_fmt", "rgba64"])
        .args(["-r", &format!("{}", fps)])
        .args(["-s", &format!("{}x{}", w, h)])
        .args(["-color_trc", "iec61966-2-1"])
        .args(["-i", "-"]);

    if has_audio && !audio_args.is_empty() {
        cmd.args(["-i", &wav_name]);
        cmd.args(audio_args);
    }

    if !video_args.is_empty() {
        cmd.args(video_args);
    }

    cmd.arg(format!("{}.mp4", file_stem));

    let mut child = cmd.spawn()?;

    let mut stdin = child.stdin.take().unwrap();
    let mut second_img = false;

    {
        let mut s = state.lock().unwrap();
        s.num_frames = frames.end - frames.start;
        s.cur_frame = 0;
        if s.abort {
            return Ok(());
        }
    }
    let (upload_buffer, _) = recv.recv().unwrap();
    compute.process(
        Some(upload_buffer),
        second_img as usize,
        pc,
        Some((output_buffers[second_img as usize].clone(), resize)),
    );
    second_img = !second_img;
    for _ in frames {
        {
            let mut s = state.lock().unwrap();
            s.cur_frame += 1;
            if s.abort {
                return Ok(());
            }
        }
        let Ok((upload_buffer, _)) = recv.recv() else {
            break;
        };
        compute.wait();
        compute.process(
            Some(upload_buffer),
            second_img as usize,
            pc,
            Some((output_buffers[second_img as usize].clone(), resize)),
        );
        second_img = !second_img;
        stdin.write_all(&output_buffers[second_img as usize].read().unwrap())?;
    }
    compute.wait();
    second_img = !second_img;
    stdin.write_all(&output_buffers[second_img as usize].read().unwrap())?;
    drop(stdin);
    child.wait().unwrap();

    if has_audio {
        fs::remove_file(&wav_name)?;
    }
    Ok(())
}

fn create_wav(name: &str, mlv: &VideoFile) -> Result<bool> {
    let trim = mlv.trim.clone();
    let Some(wi) = mlv.wav_info else {
        return Ok(false);
    };
    let frame_len = (1_000_000.0 / mlv.fps) as i64;
    let sync_offset =
        (mlv.audio_delay + trim.start as i64 * frame_len) * wi.bytes_per_second as i64 / 1_000_000;
    let mut sync_offset = (sync_offset / wi.block_align as i64) * wi.block_align as i64;

    let size = mlv.audio_frames.iter().map(|f| f.len).sum::<usize>() as i64;
    let mut desired_size =
        ((trim.end - trim.start) as i64 * frame_len) * wi.bytes_per_second as i64 / 1_000_000;
    // Align to block.
    desired_size += (-desired_size).rem_euclid(wi.block_align as i64);
    let mut size = (size - sync_offset).min(desired_size) as usize;

    let mut vidfile = File::open(&mlv.path)?;
    let mut out = BufWriter::new(File::create(name)?);

    let (num, den) = ((mlv.fps * 1000.0) as i32, 1000);
    let ixml = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>
    <BWFXML>
    <IXML_VERSION>1.5</IXML_VERSION>
    <PROJECT>Kinochrome</PROJECT>
    <NOTE></NOTE>
    <CIRCLED>FALSE</CIRCLED>
    <BLACKMAGIC-KEYWORDS></BLACKMAGIC-KEYWORDS>
    <TAPE>1</TAPE>
    <SCENE>1</SCENE>
    <BLACKMAGIC-SHOT>1</BLACKMAGIC-SHOT>
    <TAKE>1</TAKE>
    <BLACKMAGIC-ANGLE>ms</BLACKMAGIC-ANGLE>
    <SPEED>
    <MASTER_SPEED>{num}/{den}</MASTER_SPEED>
    <CURRENT_SPEED>{num}/{den}</CURRENT_SPEED>
    <TIMECODE_RATE>{num}/{den}</TIMECODE_RATE>
    <TIMECODE_FLAG>NDF</TIMECODE_FLAG>
    </SPEED>
    </BWFXML>"
    );

    out.write_all(b"RIFF")?;
    out.write_all(&((size + 36 + 1024 + 8) as u32).to_le_bytes())?;
    out.write_all(b"WAVE")?;
    out.write_all(b"iXML")?;
    out.write_all(&(1024_u32).to_le_bytes())?;
    out.write_all(ixml.as_bytes())?;
    out.write_all(&vec![0; 1024 - ixml.len()])?;
    out.write_all(b"fmt ")?;
    out.write_all(&16_u32.to_le_bytes())?;
    wi.write_packed(&mut out)?;

    out.write_all(b"data")?;
    out.write_all(&size.to_le_bytes())?;

    let mut audio_iter = mlv.audio_frames.iter();
    let Some(&AudioFrame { pos, mut len }) = audio_iter.next() else {
        fs::remove_file(name)?;
        return Ok(false);
    };
    let mut start = 0;
    if sync_offset <= 0 {
        out.write_all(&vec![0; (-sync_offset) as usize])?;
        sync_offset = 0;
    } else {
        start = sync_offset as usize;
        sync_offset = (sync_offset as u64).saturating_sub(len as u64) as i64;
        len = len.saturating_sub(start).min(size);
        size = size.saturating_sub(len);
    }
    vidfile.seek(SeekFrom::Start(pos + start as u64))?;
    let mut payload = vec![0; len];
    vidfile.read_exact(&mut payload)?;
    out.write_all(&payload)?;

    for &AudioFrame { pos, mut len } in audio_iter {
        start = sync_offset as usize;
        sync_offset = (sync_offset as u64).saturating_sub(len as u64) as i64;
        len = len.saturating_sub(start).min(size);
        size = size.saturating_sub(len);
        vidfile.seek(SeekFrom::Start(pos + start as u64))?;
        let mut payload = vec![0; len];
        vidfile.read_exact(&mut payload)?;
        out.write_all(&payload)?;
    }

    Ok(true)
}

/// Encode videos into CinemaDNG
pub fn encode_cdngs(files: Vec<(String, VideoFile)>, state: Arc<Mutex<EncodingState>>) {
    {
        let mut s = state.lock().unwrap();
        s.running = true;
        s.num_files = files.len();
        s.cur_file = 0;
    }
    for (name, mlv) in files {
        {
            let mut s = state.lock().unwrap();
            s.cur_file += 1;
            if s.abort {
                s.abort = false;
                break;
            }
        }
        if let Err(e) = encode_cdng(&name, mlv, &state) {
            log_error(&e, &format!("exporting {name}"));
        }
    }
    state.lock().unwrap().running = false;
}

fn encode_cdng(name: &str, video: VideoFile, state: &Mutex<EncodingState>) -> Result<()> {
    let mut name = name.to_string();
    let suffix = ".MLV";
    if name.ends_with(suffix) {
        name.truncate(name.len() - suffix.len());
    }

    let Some(mut ifd) = video.raw_info.as_ref().map(rawinfo_to_ifd) else {
        return Err(Error::other("RAW info not found"));
    };
    ifd.insert(ifd::ReelName, name.as_str());
    ifd.insert(
        ifd::FrameRate,
        IfdValue::SRational((video.fps * 1000.0) as i32, 1000),
    );

    let mut vidframes = match video.frames {
        Frames::Mlv(ref frames, _) => Frames::Mlv(frames.clone(), File::open(&video.path)?),
        Frames::Dng(ref frames) => Frames::Dng(frames.clone()),
    };
    fs::create_dir_all(&name)?;

    let mut num_threads: usize = thread::available_parallelism().map(Into::into).unwrap_or(1);
    if num_threads > 2 {
        num_threads -= 2;
    }
    let (mut send, mut recv) = (
        Vec::with_capacity(num_threads),
        Vec::with_capacity(num_threads),
    );
    for _ in 0..num_threads {
        let (s, r) = sync_channel(3);
        send.push(s);
        recv.push(r);
    }

    {
        let mut s = state.lock().unwrap();
        s.num_frames = video.frames.len();
        s.cur_frame = 0;
        if s.abort {
            return Ok(());
        }
    }
    thread::scope(|s| -> Result<()> {
        let mut threads = Vec::with_capacity(num_threads);
        for r in recv {
            let ifd = ifd.clone();
            threads.push(s.spawn(|| encode_single_dng(r, &name, ifd, &video)));
        }
        for i in video.trim.clone() {
            let (payload, pan) = vidframes.read_ith(i);
            send[i % num_threads].send((payload, i, pan)).unwrap();
            {
                let mut s = state.lock().unwrap();
                s.cur_frame += 1;
                if s.abort {
                    return Ok(());
                }
            }
        }
        drop(send);
        for thread in threads {
            match thread.join() {
                Ok(result) => result?,
                Err(e) => panic::resume_unwind(e),
            }
        }
        Ok(())
    })?;

    let wav_name = format!("{}/{}.wav", &name, &name);
    create_wav(&wav_name, &video)?;
    Ok(())
}

type EncData = (Vec<u8>, usize, [u16; 2]);

fn encode_single_dng(
    recv: Receiver<EncData>,
    name: &str,
    ifd: Ifd,
    video: &VideoFile,
) -> Result<()> {
    let (width, height) = (video.width, video.height);
    let encoder = Encoder::new(
        width as u16 / 2,
        height as u16,
        Components::C2,
        Bitdepth::B14,
        Predictor::P1,
        0,
        0,
    );

    let mut output = vec![0; width * height];
    while let Ok((payload, i, pan)) = recv.recv() {
        decode_image(
            &payload,
            &mut output,
            (video.compression, video.bits_per_pixel),
            [width, height],
            &video.focus_pixels,
            pan,
        )
        .map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
        let encoded = encoder.encode(&output).unwrap();
        let len = encoded.len();
        let mut ifd = ifd.clone();
        ifd.insert(ifd::StripOffsets, IfdValue::Offsets(Arc::new(encoded)));
        ifd.insert(ifd::StripByteCounts, len as u32);

        let path = format!("{}/{}_{:06}.dng", name, name, i);
        let file = File::create(&path)?;
        let writer = BufWriter::new(file);
        DngWriter::write_dng(writer, true, FileType::Dng, vec![ifd])?;
    }
    Ok(())
}

fn rawinfo_to_ifd(ri: &RawInfo) -> Ifd {
    let mut ifd = Ifd::new(IfdType::Ifd);

    ifd.insert(ifd::DNGVersion, [1_u8, 4, 0, 0]);
    ifd.insert(ifd::NewSubfileType, 0_u16);
    ifd.insert(
        ifd::DefaultScale,
        [IfdValue::Rational(1, 1), IfdValue::Rational(1, 1)],
    );
    ifd.insert(ifd::DefaultCropOrigin, [0_u16, 0]);
    ifd.insert(ifd::Orientation, 1_u16);
    ifd.insert(ifd::PlanarConfiguration, 1_u16);
    ifd.insert(ifd::FillOrder, 1_u16);
    ifd.insert(ifd::PhotometricInterpretation, 32803_u16); // CFA
    ifd.insert(ifd::CFARepeatPatternDim, [2_u16, 2]);
    ifd.insert(ifd::CFAPattern, [0_u8, 1, 1, 2]);
    ifd.insert(ifd::SamplesPerPixel, 1_u16);

    ifd.insert(ifd::Compression, 7_u16);

    let (width, height) = (ri.res_x, ri.res_y);
    ifd.insert(ifd::ImageWidth, width);
    ifd.insert(ifd::ImageLength, height);
    ifd.insert(ifd::DefaultCropSize, [width, height]);
    ifd.insert(ifd::RowsPerStrip, height);
    ifd.insert(ifd::BitsPerSample, ri.bits_per_pixel as u16);
    ifd.insert(ifd::BlackLevel, ri.black_level as u16);
    ifd.insert(ifd::WhiteLevel, ri.white_level as u16);

    ifd.insert(ifd::ColorMatrix1, ri.color_matrix.map(frac_to_ifdv));
    ifd.insert(ifd::CalibrationIlluminant1, 21_u16); // D65

    ifd
}

fn frac_to_ifdv(f: Fraction) -> IfdValue {
    IfdValue::SRational(f.numerator, f.denominator as i32)
}

/// Read all TOML-serialized recipes from the directory
pub fn read_recipes(directory: &str) -> Result<(Vec<Recipe>, Vec<Recipe>)> {
    let mut video_recipes = Vec::new();
    let mut audio_recipes = Vec::new();
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        if !entry.file_type()?.is_file()
            || !entry
                .path()
                .extension()
                .unwrap_or_default()
                .eq_ignore_ascii_case("toml")
        {
            continue;
        }
        let recipe: Recipe = match toml::from_str(&read_to_string(entry.path())?) {
            Ok(recipe) => recipe,
            Err(e) => {
                log_error(&e, &format!("parsing the recipe {:?}", entry.path()));
                continue;
            }
        };
        match recipe.kind {
            RecipeKind::Video => {
                video_recipes.push(recipe);
            }
            RecipeKind::Audio => {
                audio_recipes.push(recipe);
            }
        }
    }
    Ok((video_recipes, audio_recipes))
}

fn deserialize_shell_args<'de, D>(deserializer: D) -> std::result::Result<Vec<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let s: String = Deserialize::deserialize(deserializer)?;
    Ok(shlex::split(&s).unwrap_or_default())
}

impl Param {
    /// Produce the command line arguments from the parameter
    pub fn to_args(&self) -> [String; 2] {
        [self.arg.clone(), (&self.value).into()]
    }
}

impl From<&ParamValue> for String {
    fn from(p: &ParamValue) -> String {
        match p {
            ParamValue::Int { value, .. } => format!("{value}"),
            ParamValue::Float { value, .. } => format!("{value}"),
            ParamValue::Discrete { options, index } => options[*index][0].clone(),
        }
    }
}
