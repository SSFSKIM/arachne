//! What every drawing test shares: the compositor it draws with, chosen by the run's launch mode,
//! and fixture pictures. Test code is the only reader of the mode.
//!
//! - `ARACHNE_GPU_TEST_MODE=software` (CI): the testing door; asserts the device is of CPU type.
//! - `ARACHNE_GPU_TEST_MODE=hardware` with `ARACHNE_GPU_TEST_PIN` (a JSON object with `adapter`,
//!   `backend`, `driver` and `wgpu`): the ordinary door; asserts the device is no software adapter
//!   and its pin is the one given.
//!
//! With no mode set every drawing test fails.
//!
//! The probes that draw frames of a frame's size draw at `ARACHNE_GPU_TEST_SIZE`
//! (`<width>x<height>`, at least 480 x 270), or at 480 x 270 when it is unset: CI's size, and a
//! GPU host's second size, 1920x1080. Their closed-form bounds hold at any size.

#![allow(dead_code)]

use std::process::{Command, Output};
use std::sync::{Once, OnceLock};

use arachne_gpu::{Compositor, NoGpu, OpenError, Pin};

pub const MODE: &str = "ARACHNE_GPU_TEST_MODE";
pub const PIN: &str = "ARACHNE_GPU_TEST_PIN";
pub const SIZE: &str = "ARACHNE_GPU_TEST_SIZE";

/// The size frame-sized probes draw at when the run names none.
pub const DEFAULT_SIZE: (u32, u32) = (480, 270);

#[derive(Debug, PartialEq, Eq)]
pub enum Mode {
    Software,
    Hardware(Pin),
}

/// The launch mode, from the values of `MODE` and `PIN`.
pub fn mode_from(mode: Option<&str>, pin: Option<&str>) -> Result<Mode, String> {
    match mode {
        Some("software") => Ok(Mode::Software),
        Some("hardware") => {
            let pin = pin.ok_or(format!("hardware mode needs {PIN}"))?;
            let value: serde_json::Value =
                serde_json::from_str(pin).map_err(|e| format!("{PIN} is not JSON: {e}"))?;
            let field = |name: &str| {
                value[name]
                    .as_str()
                    .map(String::from)
                    .ok_or(format!("{PIN} has no string `{name}`"))
            };
            Ok(Mode::Hardware(Pin {
                adapter: field("adapter")?,
                backend: field("backend")?,
                driver: field("driver")?,
                wgpu: field("wgpu")?,
            }))
        }
        Some(other) => Err(format!(
            "{MODE}={other}: the modes are software and hardware"
        )),
        None => Err(format!(
            "{MODE} is not set: drawing tests run in software mode (CI) or hardware mode (a GPU host)"
        )),
    }
}

pub fn mode() -> Mode {
    let mode = std::env::var(MODE).ok();
    let pin = std::env::var(PIN).ok();
    mode_from(mode.as_deref(), pin.as_deref()).unwrap_or_else(|e| panic!("{e}"))
}

/// The drawing size, from the value of `SIZE`: `<width>x<height>`, no smaller than 480 x 270 on
/// either side, since the probes' fixed geometry fits that frame.
pub fn size_from(value: Option<&str>) -> Result<(u32, u32), String> {
    let Some(value) = value else {
        return Ok(DEFAULT_SIZE);
    };
    let parsed = value
        .split_once('x')
        .and_then(|(w, h)| Some((w.parse::<u32>().ok()?, h.parse::<u32>().ok()?)));
    match parsed {
        Some((w, h)) if w >= DEFAULT_SIZE.0 && h >= DEFAULT_SIZE.1 => Ok((w, h)),
        Some(_) => Err(format!(
            "{SIZE}={value}: the probes draw at 480x270 or larger"
        )),
        None => Err(format!("{SIZE}={value}: not <width>x<height>")),
    }
}

/// The size frame-sized probes draw at, read once.
pub fn size() -> (u32, u32) {
    static DRAWING: OnceLock<(u32, u32)> = OnceLock::new();
    *DRAWING.get_or_init(|| {
        size_from(std::env::var(SIZE).ok().as_deref()).unwrap_or_else(|e| panic!("{e}"))
    })
}

static REPORTED: Once = Once::new();

/// A compositor through the door the launch mode names, unchecked: for tests of what a door
/// refuses.
pub fn open(width: u32, height: u32) -> Result<Compositor, OpenError> {
    match mode() {
        Mode::Software => arachne_gpu::testing::software_compositor(width, height),
        Mode::Hardware(_) => Compositor::new(width, height),
    }
}

/// The adapter failure a door's refusal holds; fails on anything else.
pub fn no_gpu(opened: Result<Compositor, OpenError>) -> NoGpu {
    match opened {
        Err(OpenError::NoGpu(no_gpu)) => no_gpu,
        Err(other) => panic!("not an adapter failure: {other}"),
        Ok(_) => panic!("a compositor, where none should open"),
    }
}

/// The compositor a drawing test draws with, its device checked against the launch mode.
pub fn compositor(width: u32, height: u32) -> Compositor {
    let compositor = match mode() {
        Mode::Software => {
            let c = arachne_gpu::testing::software_compositor(width, height)
                .unwrap_or_else(|e| panic!("software mode: {e}"));
            assert_eq!(
                c.adapter().device_type,
                "Cpu",
                "software mode drew on {}",
                c.adapter()
            );
            assert!(c.software());
            c
        }
        Mode::Hardware(pin) => {
            let c = Compositor::new(width, height).unwrap_or_else(|e| panic!("hardware mode: {e}"));
            assert!(!c.software());
            assert_ne!(c.adapter().device_type, "Cpu");
            assert_eq!(c.pin(), pin, "hardware mode drew on another pin");
            c
        }
    };
    REPORTED.call_once(|| {
        eprintln!("ADAPTER: {}", compositor.adapter());
        let (w, h) = size();
        eprintln!("SIZE: {w}x{h}");
    });
    compositor
}

/// A deterministic byte stream (a 64-bit LCG's high bits).
pub struct Lcg(pub u64);

impl Lcg {
    pub fn byte(&mut self) -> u8 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 56) as u8
    }
}

/// Valid painted RGBA: every colour value at most its alpha.
pub fn painted_picture(width: u32, height: u32, seed: u64) -> Vec<u8> {
    let mut lcg = Lcg(seed);
    let mut rgba = Vec::with_capacity((width * height * 4) as usize);
    for _ in 0..width * height {
        let a = lcg.byte();
        for _ in 0..3 {
            let c = lcg.byte();
            rgba.push(((u16::from(c) * u16::from(a)) / 255) as u8);
        }
        rgba.push(a);
    }
    rgba
}

/// One colour, painted, everywhere.
pub fn flat(width: u32, height: u32, rgba: [u8; 4]) -> Vec<u8> {
    rgba.repeat((width * height) as usize)
}

/// round(a * b / 255).
pub fn mul_un8(a: u8, b: u8) -> u8 {
    ((f64::from(a) * f64::from(b) / 255.0).round()) as u8
}

/// This test binary run again as a child, for one test only, with the environment changed: how a
/// test makes a machine with no GPU or no Vulkan loader for itself.
pub fn run_child(test: &str, env: &[(&str, &str)]) -> Output {
    let exe = std::env::current_exe().expect("the test binary");
    let mut command = Command::new(exe);
    command.args([test, "--exact", "--nocapture", "--test-threads=1"]);
    command.env("ARACHNE_GPU_TEST_CHILD", "1");
    for (name, value) in env {
        command.env(name, value);
    }
    command.output().expect("the child test runs")
}

/// True in a child started by [`run_child`].
pub fn is_child() -> bool {
    std::env::var_os("ARACHNE_GPU_TEST_CHILD").is_some()
}

/// Lavapipe's ICD manifest: a Vulkan loader shown only it sees no GPU.
pub fn software_icd() -> String {
    let dir = std::path::Path::new("/usr/share/vulkan/icd.d");
    let mut found: Vec<_> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| {
            path.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("lvp_icd") && n.ends_with(".json"))
        })
        .collect();
    found.sort();
    found
        .first()
        .unwrap_or_else(|| panic!("no lavapipe ICD manifest in {}", dir.display()))
        .display()
        .to_string()
}

/// Uploads a painted picture and returns its key.
pub fn upload(c: &mut Compositor, width: u32, height: u32, rgba: &[u8]) -> arachne_gpu::Key {
    let key = arachne_gpu::painted_key(width, height, rgba);
    c.upload_painted(key, width, height, rgba).unwrap();
    key
}

/// The picture made opaque: every alpha 255 (a painted picture stays painted).
pub fn opaque(mut rgba: Vec<u8>) -> Vec<u8> {
    for pixel in rgba.as_chunks_mut::<4>().0 {
        pixel[3] = 255;
    }
    rgba
}

/// The largest difference of any channel of any pixel.
pub fn max_difference(a: &[u8], b: &[u8]) -> u8 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| x.abs_diff(*y))
        .max()
        .unwrap_or(0)
}

/// The four bytes of pixel (`x`, `y`) of a frame `width` wide.
pub fn pixel(rgba: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
    let at = ((y * width + x) * 4) as usize;
    rgba[at..at + 4].try_into().unwrap()
}

/// Corners in CE's order for the rectangle from (`x0`, `y0`) to (`x1`, `y1`).
pub fn rectangle(x0: f64, y0: f64, x1: f64, y1: f64) -> [[f64; 2]; 4] {
    [[x0, y0], [x1, y0], [x0, y1], [x1, y1]]
}
