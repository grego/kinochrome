use crate::color_utils::Illuminant;
use crate::encoding::{ParamValue, Recipe};
use crate::state::State;

use egui::load::SizedTexture;
use egui::{
    Button, CentralPanel, Color32, ComboBox, Context, CursorIcon, DragValue, Event, Id, Image,
    ImageSource, Key, MenuBar, Modifiers, Pos2, ProgressBar, Rect, Response, Rgba, ScrollArea,
    Sense, SidePanel, Slider, TopBottomPanel, Ui, Vec2, Window, style,
};

use std::f32::consts::PI;
use std::time::Instant;

/// GUI layout
pub fn layout(s: &mut State, ctx: &Context) {
    TopBottomPanel::top("top panel").show(ctx, |ui| {
        MenuBar::new().ui(ui, |ui| {
            ui.menu_button("File", |ui| {
                if ui.button("Open").on_hover_text("Ctrl + O").clicked() {
                    s.open_dialog.pick_file();
                }
                if ui.button("Save").on_hover_text("Ctrl + S").clicked() {
                    s.open_dialog.save_file();
                }
                ui.separator();

                if ui.button("Import").on_hover_text("Ctrl + I").clicked() {
                    s.import_dialog.pick_multiple();
                }
                if ui.button("Export").on_hover_text("Ctrl + E").clicked() {
                    s.encoding_dialog.shown = true;
                    s.encoding_dialog.resize = s.extent;
                    s.encoding_dialog.fps = 1_000_000.0 / s.ideal_frame_len;
                }
                if ui.button("Export Cinema DNG").clicked() {
                    s.encode_cdng = true;
                    s.import_dialog.pick_directory();
                }
            });
            ui.menu_button("Edit", |ui| {
                if ui
                    .add_enabled(s.has_undo(), Button::new("Undo"))
                    .on_hover_text("Ctrl + Z")
                    .clicked()
                {
                    s.undo();
                }
                if ui
                    .add_enabled(s.has_redo(), Button::new("Redo"))
                    .on_hover_text("Ctrl + Y")
                    .clicked()
                {
                    s.redo();
                }
            });
            ui.menu_button("Info", |ui| {
                if ui.button("About").clicked() {
                    s.about_dialog_shown = true;
                }
            });
        });
    });

    SidePanel::left("File Panel").show(ctx, |ui| {
        ScrollArea::vertical().show(ui, |ui| {
            let mut last_selected = 0;
            let mut select_range = None;
            for (i, (filename, video)) in s.files.iter_mut().enumerate() {
                let flabel = ui.add(Button::selectable(video.selected, filename));

                if flabel.clicked() {
                    if ui.input(|i| i.modifiers.shift) {
                        select_range = Some((last_selected, i));
                    } else {
                        video.selected = !video.selected;
                    }
                }

                if flabel.double_clicked() {
                    s.changed_file = Some(filename.clone());
                }

                if video.selected {
                    last_selected = i;
                }
            }
            if let Some((i, j)) = select_range {
                for (_, video) in s.files.iter_mut().skip(i + 1).take(j - i) {
                    video.selected = true;
                }
            }

            let n = s.files_being_imported;
            if n > 0 {
                ui.separator();
                ui.label(format!(
                    "Importing {n} file{}",
                    if n != 1 { "s" } else { "" }
                ));
            }
        });
    });

    SidePanel::right("Side Panel")
        .default_width(150.0)
        .show(ctx, |ui| {
            ui.heading("corrections");
            ui.add(
                Slider::new(&mut s.pc.exposure, -5.0..=5.0)
                    .text("exposure")
                    .step_by(0.1),
            );
            ComboBox::from_label("illuminant")
                .selected_text(s.color_params.illuminant.description())
                .show_ui(ui, |ui| {
                    for illuminant in Illuminant::defaults() {
                        ui.selectable_value(
                            &mut s.color_params.illuminant,
                            illuminant,
                            illuminant.description(),
                        );
                    }
                });
            match s.color_params.illuminant {
                Illuminant::A => {}
                Illuminant::D(ref mut b) => {
                    ui.add(
                        Slider::new(b, 3000..=7000)
                            .text("temerature (K)")
                            .step_by(1.0),
                    );
                }
                Illuminant::Blackbody(ref mut b) => {
                    ui.add(
                        Slider::new(b, 3000..=7000)
                            .text("temerature (K)")
                            .step_by(1.0),
                    );
                }
                Illuminant::Custom(ref mut ch) => {
                    ui.add(
                        Slider::new(&mut ch[1], -PI..=PI)
                            .text("hue")
                            .custom_formatter(|n, _| format!("{:.3}°", n * 180.0 / PI as f64))
                            .step_by(0.01),
                    );
                    ui.add(
                        Slider::new(&mut ch[0], 0.0..=0.3)
                            .text("chroma")
                            .step_by(0.01),
                    );
                }
            }
            ui.horizontal(|ui| {
                let [r, g, b] = s.color_params.illuminant.srgb();
                color_rect(ui, Rgba::from_rgb(r, g, b).into());
                if ui.add(Button::new("Pick white")).clicked() {
                    s.picker_mode = true;
                    ctx.set_cursor_icon(CursorIcon::Crosshair);
                }
            });
            ui.separator();

            ui.heading("saturation");
            ui.add_enabled_ui(!s.color_params.monochrome, |ui| {
                ui.add(
                    Slider::new(&mut s.color_params.saturation_global, -0.5..=0.5)
                        .text("global")
                        .step_by(0.01),
                );
                ui.add(
                    Slider::new(&mut s.pc.saturation_shd, -0.5..=0.5)
                        .text("shadows")
                        .step_by(0.01),
                );
                ui.add(
                    Slider::new(&mut s.pc.saturation_mid, -0.5..=0.5)
                        .text("midtones")
                        .step_by(0.01),
                );
                ui.add(
                    Slider::new(&mut s.pc.saturation_hig, -0.5..=0.5)
                        .text("highlights")
                        .step_by(0.01),
                );
            });
            ui.checkbox(&mut s.color_params.monochrome, "Monochromatic");
            ui.separator();

            ui.heading("tone mapping");
            ui.add(
                Slider::new(&mut s.pc.white_re, 1.0..=7.0)
                    .text("white exposure")
                    .step_by(0.01),
            );
            ui.add(
                Slider::new(&mut s.pc.black_re, -12.0..=-4.0)
                    .text("black exposure")
                    .step_by(0.01),
            );
            ui.add(
                Slider::new(&mut s.pc.contrast, 0.0..=2.0)
                    .text("contrast")
                    .step_by(0.01),
            );
        });

    TopBottomPanel::bottom("Bottom panel")
        .show_separator_line(false)
        .show(ctx, |ui| {
            if s.frames_len == 0 {
                ui.disable();
            }
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing = Vec2::ZERO;
                //ui.spacing_mut().window_margin = MarginF32::same(-sw / 2.0).into();
                let av = ui.available_size();

                let f = s.frames_len as f32;
                if s.trim.start > 0 {
                    ui.allocate_exact_size(
                        Vec2 {
                            x: s.trim.start as f32 * av.x / f,
                            y: av.y,
                        },
                        Sense::empty(),
                    );
                }
                if s.trim.end - s.trim.start > 0 {
                    let (rect, _) = ui.allocate_exact_size(
                        Vec2 {
                            x: (s.trim.end - s.trim.start) as f32 * av.x / f,
                            y: av.y / 2.0,
                        },
                        Sense::empty(),
                    );
                    if s.trim.start != 0 || s.trim.end != s.frames_len {
                        ui.painter()
                            .rect_filled(rect, 0.0, Rgba::from_luminance_alpha(0.5, 1.0));
                    }
                }
            });
            ui.horizontal(|ui| {
                ui.spacing_mut().slider_width = ui.available_width();
                ui.add(
                    Slider::new(&mut s.frame_number, 0..=(s.frames_len.saturating_sub(1)))
                        .handle_shape(style::HandleShape::Rect { aspect_ratio: 0.35 })
                        .integer(),
                );
            });
            ui.horizontal(|ui| {
                let icon = if s.paused { "⏵" } else { "⏸" };
                if ui
                    .button(icon)
                    .on_hover_text(if s.paused {
                        "Play (Space)"
                    } else {
                        "Pause (Space)"
                    })
                    .clicked()
                {
                    s.paused = !s.paused;
                    s.first_start = Instant::now();
                    s.vid_frame_start = Instant::now();
                }

                if ui.button("{").on_hover_text("Trim in (I)").clicked() {
                    s.mark_in(s.frame_number);
                }
                if ui.button("}").on_hover_text("Trim out (O)").clicked() {
                    s.mark_out(s.frame_number);
                }

                let [cm, cs, cf] = timestamp(s.frame_number, s.ideal_frame_len);
                let [wm, ws, wf] = timestamp(s.frames_len, s.ideal_frame_len);
                ui.label(format!(
                    "{cm:02} : {cs:02} : {cf:02} / {wm:02} : {ws:02} : {wf:02}"
                ))
            });
        });

    CentralPanel::default().show(ctx, |ui| {
        let avs = ui.available_size();
        let [width, height] = s.extent;

        if width == 0 || height == 0 {
            return;
        }
        let Some(image_id) = s.current_img else {
            return;
        };

        let ratio = width as f32 / height as f32;
        let (w, h) = if ratio < avs.x / avs.y {
            (avs.y * ratio, avs.y)
        } else {
            (avs.x, avs.x / ratio)
        };

        let response = ui
            .add(
                Image::new(ImageSource::Texture(SizedTexture::new(image_id, [w, h])))
                    .uv(Rect::from_center_size(s.center, Vec2::new(s.zoom, s.zoom))),
            )
            .interact(if s.picker_mode {
                Sense::CLICK | Sense::HOVER
            } else {
                Sense::DRAG | Sense::HOVER
            });
        let rect = response.interact_rect;
        let hover_pos = response.hover_pos().map(|p| get_relative_pos(p, rect));
        let origin: Pos2 = [0.5, 0.5].into();
        if response.clicked() {
            if let Some(pos) = response.interact_pointer_pos() {
                if s.picker_mode {
                    let Pos2 { x, y } = s.center + s.zoom * (get_relative_pos(pos, rect) - origin);
                    s.picked_point = Some([x, y]);
                }
            }
        } else if response.dragged() {
            let mut delta = response.drag_delta();
            delta.x /= rect.width();
            delta.y /= rect.height();
            s.center -= delta * s.zoom;
            let z = s.zoom / 2.0;
            s.center = s.center.clamp([z, z].into(), [1.0 - z, 1.0 - z].into());
        }

        let zoom_unit: f32 = 1.05;
        let zoom_delta = ui.input(|i| {
            i.events.iter().find_map(|e| match e {
                Event::MouseWheel { delta, .. } => Some(zoom_unit.powf(-delta.y)),
                Event::Zoom(delta) => Some(*delta),
                _ => None,
            })
        });
        if let Some(delta) = zoom_delta {
            let old_zoom = s.zoom;
            s.zoom = (s.zoom * delta).clamp(0.05, 1.0);
            if let Some(hp) = hover_pos {
                let shift = Pos2::default() - s.center;
                let old = (hp - origin.to_vec2()) * old_zoom + shift;
                let new = (hp - origin.to_vec2()) * s.zoom + shift;
                s.center += old - new;
                let z = s.zoom / 2.0;
                s.center = s.center.clamp([z, z].into(), [1.0 - z, 1.0 - z].into());
            }
        }
    });

    let es = s.encoding_state.lock().unwrap().clone();
    if es.running {
        let id = Id::new("Export");
        let title = format!("Exporting file {}/{}", es.cur_file, es.num_files);
        Window::new(&title)
            .id(id)
            .collapsible(true)
            .show(ctx, |ui| {
                ui.add(
                    ProgressBar::new(es.cur_frame as f32 / es.num_frames as f32).show_percentage(),
                );
            });
    } else if s.encoding_dialog.shown {
        Window::new("Export videos")
            .open(&mut s.encoding_dialog.shown)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.checkbox(&mut s.encoding_dialog.allow_resize, "Resize");
                    if !s.encoding_dialog.allow_resize {
                        ui.disable();
                    }
                    let [w, h] = s.encoding_dialog.resize;
                    let [ew, eh] = s.extent;
                    let ratio = ew as f64 / eh as f64;
                    ui.add(DragValue::new(&mut s.encoding_dialog.resize[0]));
                    ui.label("width");
                    ui.add(DragValue::new(&mut s.encoding_dialog.resize[1]));
                    ui.label("height");

                    let [nw, nh] = s.encoding_dialog.resize;
                    if nw != w {
                        s.encoding_dialog.resize[1] = (nw as f64 / ratio) as u32;
                    } else if nh != h {
                        s.encoding_dialog.resize[0] = (nh as f64 * ratio) as u32;
                    }
                });
                ui.horizontal(|ui| {
                    ui.checkbox(&mut s.encoding_dialog.allow_fps, "Change FPS");
                    if !s.encoding_dialog.allow_fps {
                        ui.disable();
                    }
                    ui.add(DragValue::new(&mut s.encoding_dialog.fps).range(0.0..=120.0));
                    ui.label("FPS");
                });
                ui.separator();

                encoding_form(
                    ui,
                    &mut s.encoding_dialog.video_recipes,
                    &mut s.encoding_dialog.current_video_recipe,
                    "video codec",
                );
                ui.separator();
                encoding_form(
                    ui,
                    &mut s.encoding_dialog.audio_recipes,
                    &mut s.encoding_dialog.current_audio_recipe,
                    "audio codec",
                );
                if ui.add(Button::new("Export")).clicked() {
                    s.encode_cdng = false;
                    s.import_dialog.pick_directory();
                }
            });
    }

    if s.about_dialog_shown {
        Window::new("About")
            .open(&mut s.about_dialog_shown)
            .show(ctx, |ui| {
                ui.heading(format!(
                    "{} {}",
                    env!("CARGO_CRATE_NAME"),
                    env!("CARGO_PKG_VERSION")
                ));
                ui.label(env!("CARGO_PKG_DESCRIPTION"));
                ui.hyperlink_to("source code", env!("CARGO_PKG_REPOSITORY"));
            });
    }

    if ctx.input_mut(|i| i.consume_key(Modifiers::COMMAND, Key::S)) {
        s.open_dialog.save_file();
    }
    if ctx.input_mut(|i| i.consume_key(Modifiers::COMMAND, Key::O)) {
        s.open_dialog.save_file();
    }
    if ctx.input_mut(|i| i.consume_key(Modifiers::COMMAND, Key::I)) {
        s.import_dialog.pick_multiple();
    }
    if ctx.input_mut(|i| i.consume_key(Modifiers::COMMAND, Key::E)) {
        s.encoding_dialog.shown = true;
        s.encoding_dialog.resize = s.extent;
        s.encoding_dialog.fps = 1_000_000.0 / s.ideal_frame_len;
    }
    if ctx.input(|i| i.key_pressed(Key::Space)) {
        s.paused = !s.paused;
        s.frame_start = Instant::now();
        s.vid_frame_start = Instant::now();
    }

    if ctx.input_mut(|i| {
        i.consume_key(Modifiers::COMMAND, Key::Y)
            || i.consume_key(Modifiers::COMMAND | Modifiers::SHIFT, Key::Z)
    }) {
        s.redo();
    }
    if ctx.input_mut(|i| i.consume_key(Modifiers::COMMAND, Key::Z)) {
        s.undo();
    }

    if ctx.input_mut(|i| i.consume_key(Modifiers::COMMAND, Key::Plus)) {
        s.zoom = (s.zoom / 1.1).clamp(0.05, 1.0);
    }
    if ctx.input_mut(|i| i.consume_key(Modifiers::COMMAND, Key::Minus)) {
        s.zoom = (s.zoom * 1.1).clamp(0.05, 1.0);
    }
    if ctx.input_mut(|i| i.key_pressed(Key::R)) {
        s.zoom = 1.0;
        s.center = [0.5, 0.5].into();
    }

    if ctx.input_mut(|i| i.key_pressed(Key::ArrowLeft) || i.key_pressed(Key::Comma)) {
        s.frame_number = s.frame_number.saturating_sub(1);
    }
    if ctx.input_mut(|i| i.key_pressed(Key::ArrowRight) || i.key_pressed(Key::Period)) {
        s.frame_number = (s.frame_number + 1).min(s.frames_len.saturating_sub(1));
    }

    if ctx.input_mut(|i| i.key_pressed(Key::I)) {
        s.mark_in(s.frame_number);
    }
    if ctx.input_mut(|i| i.key_pressed(Key::O)) {
        s.mark_out(s.frame_number);
    }
}

/// A form for displaying and modifying the ffmpeg encoding parameters
fn encoding_form(ui: &mut Ui, recipes: &mut [Recipe], index: &mut Option<usize>, label: &str) {
    ComboBox::from_label(label)
        .selected_text(index.map(|i| recipes[i].name.as_str()).unwrap_or_default())
        .show_ui(ui, |ui| {
            for (i, recipe) in recipes.iter().enumerate() {
                ui.selectable_value(index, Some(i), &recipe.name);
            }
        });
    if let Some(i) = index {
        let current = &mut recipes[*i];
        let len = current.params.len();
        for i in 0..len {
            let param = &mut current.params[i];
            match param.value {
                ParamValue::Int {
                    min,
                    max,
                    ref mut value,
                } => {
                    ui.add(Slider::new(value, min..=max).text(&param.name).step_by(1.0));
                }
                ParamValue::Float {
                    min,
                    max,
                    ref mut value,
                } => {
                    ui.add(
                        Slider::new(value, min..=max)
                            .text(&param.name)
                            .step_by(0.01),
                    );
                }
                ParamValue::Discrete {
                    ref options,
                    ref mut index,
                } => {
                    ComboBox::from_label(&param.name)
                        .selected_text(&options[*index][0])
                        .show_ui(ui, |ui| {
                            for (i, option) in options.iter().enumerate() {
                                ui.selectable_value(index, i, &option[0]);
                            }
                        });
                }
            }
        }
    }
}

/// Draw a color rectangle
fn color_rect(ui: &mut Ui, color: Color32) -> Response {
    let size = ui.spacing().interact_size;
    let (rect, response) = ui.allocate_exact_size(size, Sense::click());

    if ui.is_rect_visible(rect) {
        ui.painter().rect_filled(rect, 0.0, color);
    }

    response
}

/// Get position of an event relative to an interact rectangle
fn get_relative_pos(Pos2 { x, y }: Pos2, r: Rect) -> Pos2 {
    let x = ((x - r.min.x) / r.width()).clamp(0.0, 1.0);
    let y = ((y - r.min.y) / r.height()).clamp(0.0, 1.0);
    Pos2 { x, y }
}

/// Calculate the timestamp from the number of frames and the frame length in microseconds.
fn timestamp(frames: usize, frame_len: f32) -> [u16; 3] {
    let time = frames as f32 * frame_len;
    let whole_seconds = (time / 1_000_000.0) as u16;
    let m = whole_seconds / 60;
    let s = whole_seconds % 60;
    let f = ((time - whole_seconds as f32 * 1_000_000.0) / frame_len) as u16;
    [m, s, f]
}
