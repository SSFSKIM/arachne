//! The read-back benchmark: how long a frame of the reference and heavy stacks takes, drawn and
//! read back with the one-frame copy `Compositor::draw` makes, and with a pipelined copy that takes
//! frame n out while frame n + 1 draws, through a small ring of buffers ([`Pipeline`]). Also the
//! time of one frame from a warm compositor after every picture of the stack is uploaded anew, as
//! when a scene is asked for a frame at a new time.
//!
//! The pipelined copy is measured here and is no part of `Compositor`'s drawing API.

use std::collections::VecDeque;
use std::path::Path;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use super::stacks::{self, Stack};
use super::{Error, Open, json};
use crate::{AdapterSummary, Compositor, DrawError, Frame, Pin};

/// Frames drawn through a ring of read-back buffers: up to `depth` frames are in flight, and the
/// oldest is copied out while the ones after it draw. Frames come out in the order they went in.
pub struct Pipeline<'c> {
    compositor: &'c mut Compositor,
    ring: Vec<wgpu::Buffer>,
    free: Vec<usize>,
    in_flight: VecDeque<(usize, wgpu::SubmissionIndex)>,
}

impl<'c> Pipeline<'c> {
    /// A ring of `depth` read-back buffers (at least 2) for `compositor`'s frames.
    pub fn new(compositor: &'c mut Compositor, depth: usize) -> Self {
        assert!(depth >= 2, "a pipeline needs at least two buffers");
        let ring = (0..depth).map(|_| compositor.readback_buffer()).collect();
        Pipeline {
            compositor,
            ring,
            free: (0..depth).rev().collect(),
            in_flight: VecDeque::with_capacity(depth),
        }
    }

    /// Submits `frame`. With every buffer in flight, first takes the oldest frame out and returns
    /// it.
    pub fn push(&mut self, frame: &Frame) -> Result<Option<Vec<u8>>, DrawError> {
        let out = if self.free.is_empty() {
            self.pop()?
        } else {
            None
        };
        let slot = self.free.pop().expect("a buffer is free");
        let submission = self.compositor.submit(frame, &self.ring[slot]);
        match submission {
            Ok(submission) => self.in_flight.push_back((slot, submission)),
            Err(refused) => {
                self.free.push(slot);
                return Err(refused);
            }
        }
        Ok(out)
    }

    /// Takes the oldest frame in flight out, waiting for it; `None` with none in flight.
    pub fn pop(&mut self) -> Result<Option<Vec<u8>>, DrawError> {
        let Some((slot, submission)) = self.in_flight.pop_front() else {
            return Ok(None);
        };
        let rgba = self.compositor.collect(&self.ring[slot], submission);
        self.free.push(slot);
        rgba.map(Some)
    }
}

/// How many frames, and through how deep a ring.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Options {
    /// Frames timed, after the warm-up.
    pub frames: usize,
    /// Frames drawn and not timed, first.
    pub warmup: usize,
    /// Read-back buffers in the pipelined copy's ring.
    pub ring: usize,
    /// Frames timed after a new upload of every picture of the stack.
    pub uploads: usize,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            frames: 300,
            warmup: 30,
            ring: 3,
            uploads: 30,
        }
    }
}

/// The mean and the 95th percentile (nearest rank) of a list of times, in milliseconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Summary {
    pub samples: usize,
    pub mean_ms: f64,
    pub p95_ms: f64,
}

impl Summary {
    pub fn of(times: &[Duration]) -> Summary {
        let mut ms: Vec<f64> = times.iter().map(|t| t.as_secs_f64() * 1e3).collect();
        ms.sort_by(f64::total_cmp);
        let samples = ms.len();
        let (mean_ms, p95_ms) = if samples == 0 {
            (f64::NAN, f64::NAN)
        } else {
            let rank = (0.95 * samples as f64).ceil() as usize;
            (ms.iter().sum::<f64>() / samples as f64, ms[rank.max(1) - 1])
        };
        Summary {
            samples,
            mean_ms,
            p95_ms,
        }
    }

    fn json(&self) -> String {
        json::object(&[
            ("samples", self.samples.to_string()),
            ("mean_ms", json::number(self.mean_ms)),
            ("p95_ms", json::number(self.p95_ms)),
        ])
    }
}

/// When one part of a stack's timing ran, in seconds since the Unix epoch, so that a GPU monitor
/// sampled beside it can be read against it.
#[derive(Clone, Debug, PartialEq)]
pub struct Phase {
    pub name: &'static str,
    pub started: f64,
    pub ended: f64,
}

/// One stack's times.
#[derive(Clone, Debug, PartialEq)]
pub struct StackTiming {
    pub name: String,
    pub width: u32,
    pub height: u32,
    /// Each frame drawn and copied out before the next is drawn (`Compositor::draw`).
    pub one_frame: Summary,
    /// The interval between frames taken out of the pipelined copy with its ring full, each taken
    /// out as the next is submitted: its steady state.
    pub pipelined: Summary,
    /// Every picture of the stack uploaded anew to a warm compositor (the texture writes are
    /// queued here and reach the device with the frame's submission).
    pub upload: Summary,
    /// That upload and the frame drawn and read back: one frame at a new time.
    pub upload_and_frame: Summary,
    pub phases: Vec<Phase>,
}

/// The whole run: the device and every stack's times.
#[derive(Clone, Debug, PartialEq)]
pub struct Report {
    pub pin: Pin,
    pub adapter: AdapterSummary,
    pub software: bool,
    pub options: Options,
    pub stacks: Vec<StackTiming>,
}

fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |t| t.as_secs_f64())
}

fn differs(name: &str, how: &str) -> Error {
    format!("{name}: {how} read back other bytes than the first draw").into()
}

/// Times one stack on `compositor`: `build(0)` is the stack, and `build(n)` for n from 1 its new
/// pictures. Every frame drawn is checked against the first draw, outside the timed window, except
/// the pipelined copy's timed frames, of which only the last is (a comparison there would be
/// timed).
pub fn time_stack(
    compositor: &mut Compositor,
    build: &dyn Fn(u64) -> Stack,
    options: &Options,
) -> Result<StackTiming, Error> {
    let stack = build(0);
    let name = stack.name.clone();
    stack.upload(compositor)?;
    let mut phases = Vec::new();

    let started = now();
    let expected = compositor.draw(&stack.frame)?;
    for _ in 0..options.warmup {
        if compositor.draw(&stack.frame)? != expected {
            return Err(differs(&name, "the one-frame copy"));
        }
    }
    let mut times = Vec::with_capacity(options.frames);
    for _ in 0..options.frames {
        let start = Instant::now();
        let rgba = compositor.draw(&stack.frame)?;
        times.push(start.elapsed());
        if rgba != expected {
            return Err(differs(&name, "the one-frame copy"));
        }
    }
    let one_frame = Summary::of(&times);
    phases.push(Phase {
        name: "one_frame",
        started,
        ended: now(),
    });

    // The ring is filled and warmed, and each warm-up frame compared, before the timed window;
    // in it, every frame taken out comes with the next one submitted behind it, so the ring stays
    // full. The last timed frame is compared, and the ring drained and compared, after it.
    let started = now();
    let mut intervals = Vec::with_capacity(options.frames);
    {
        let mut pipeline = Pipeline::new(compositor, options.ring);
        for _ in 0..options.ring {
            if pipeline.push(&stack.frame)?.is_some() {
                unreachable!("a ring gives nothing back before it is full");
            }
        }
        /// One frame through a full ring: the oldest taken out, the next submitted.
        fn full(pipeline: &mut Pipeline<'_>, frame: &Frame) -> Result<Vec<u8>, Error> {
            Ok(pipeline
                .push(frame)?
                .expect("a full ring gives one frame back for each pushed"))
        }
        for _ in 0..options.warmup {
            if full(&mut pipeline, &stack.frame)? != expected {
                return Err(differs(&name, "the pipelined copy"));
            }
        }
        let mut last_frame = None;
        let mut last = Instant::now();
        for _ in 0..options.frames {
            let rgba = full(&mut pipeline, &stack.frame)?;
            let at = Instant::now();
            intervals.push(at - last);
            last = at;
            last_frame = Some(rgba);
        }
        if last_frame.is_some_and(|rgba| rgba != expected) {
            return Err(differs(&name, "the pipelined copy"));
        }
        while let Some(rgba) = pipeline.pop()? {
            if rgba != expected {
                return Err(differs(&name, "the pipelined copy"));
            }
        }
    }
    let pipelined = Summary::of(&intervals);
    phases.push(Phase {
        name: "pipelined",
        started,
        ended: now(),
    });

    let started = now();
    let mut uploads = Vec::with_capacity(options.uploads);
    let mut frames = Vec::with_capacity(options.uploads);
    let mut current = stack;
    for variant in 1..=options.uploads as u64 {
        let next = build(variant);
        current.release(compositor);
        let start = Instant::now();
        next.upload(compositor)?;
        uploads.push(start.elapsed());
        compositor.draw(&next.frame)?;
        frames.push(start.elapsed());
        current = next;
    }
    current.release(compositor);
    phases.push(Phase {
        name: "new_upload",
        started,
        ended: now(),
    });

    Ok(StackTiming {
        name,
        width: compositor.width(),
        height: compositor.height(),
        one_frame,
        pipelined,
        upload: Summary::of(&uploads),
        upload_and_frame: Summary::of(&frames),
        phases,
    })
}

/// Times the reference and the heavy stack at each size, one compositor per size, each opened by
/// `open`. Fails if two compositors are on different devices.
pub fn run(open: &mut Open<'_>, sizes: &[(u32, u32)], options: &Options) -> Result<Report, Error> {
    if options.ring < 2 {
        return Err("the pipelined copy's ring needs at least two buffers".into());
    }
    let mut device: Option<(Pin, AdapterSummary, bool)> = None;
    let mut timings = Vec::new();
    for &(width, height) in sizes {
        let mut compositor = open(width, height)?;
        let this = (
            compositor.pin(),
            compositor.adapter(),
            compositor.software(),
        );
        if let Some(first) = device.as_ref().filter(|first| **first != this) {
            return Err(format!("timed on two devices: {first:?} and {this:?}").into());
        }
        if device.is_none() {
            device = Some(this);
        }
        let reference = move |variant| stacks::reference(width, height, variant);
        let heavy = move |variant| stacks::heavy(width, height, variant);
        for build in [&reference as &dyn Fn(u64) -> Stack, &heavy] {
            timings.push(time_stack(&mut compositor, build, options)?);
        }
    }
    let (pin, adapter, software) = device.ok_or("no size was given")?;
    Ok(Report {
        pin,
        adapter,
        software,
        options: *options,
        stacks: timings,
    })
}

/// The report as JSON.
pub fn to_json(report: &Report) -> String {
    let stacks: Vec<String> = report
        .stacks
        .iter()
        .map(|s| {
            let phases: Vec<String> = s
                .phases
                .iter()
                .map(|p| {
                    json::object(&[
                        ("name", json::string(p.name)),
                        ("started", json::number(p.started)),
                        ("ended", json::number(p.ended)),
                    ])
                })
                .collect();
            json::object(&[
                ("name", json::string(&s.name)),
                ("width", s.width.to_string()),
                ("height", s.height.to_string()),
                ("one_frame", s.one_frame.json()),
                ("pipelined", s.pipelined.json()),
                ("upload", s.upload.json()),
                ("upload_and_frame", s.upload_and_frame.json()),
                ("phases", format!("[{}]", phases.join(", "))),
            ])
        })
        .collect();
    let options = &report.options;
    let top = json::object(&[
        ("frame_set", stacks::VERSION.to_string()),
        ("pin", json::pin(&report.pin)),
        ("adapter", json::adapter(&report.adapter)),
        ("software", report.software.to_string()),
        (
            "options",
            json::object(&[
                ("frames", options.frames.to_string()),
                ("warmup", options.warmup.to_string()),
                ("ring", options.ring.to_string()),
                ("uploads", options.uploads.to_string()),
            ]),
        ),
        ("stacks", json::array(&stacks)),
    ]);
    format!("{top}\n")
}

/// One line per stack, for a log.
pub fn lines(report: &Report) -> Vec<String> {
    let ring = report.options.ring;
    report
        .stacks
        .iter()
        .map(|s| {
            format!(
                "{}: one-frame copy mean {:.3} ms p95 {:.3} ms; pipelined (ring of {ring}) mean \
                 {:.3} ms p95 {:.3} ms; new upload and frame mean {:.3} ms p95 {:.3} ms (upload \
                 alone mean {:.3} ms)",
                s.name,
                s.one_frame.mean_ms,
                s.one_frame.p95_ms,
                s.pipelined.mean_ms,
                s.pipelined.p95_ms,
                s.upload_and_frame.mean_ms,
                s.upload_and_frame.p95_ms,
                s.upload.mean_ms,
            )
        })
        .collect()
}

/// Writes the report as JSON to `path`.
pub fn write(report: &Report, path: &Path) -> Result<(), Error> {
    std::fs::write(path, to_json(report))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::software_compositor;
    use crate::{Layer, painted_key};

    #[test]
    fn the_summary_is_the_mean_and_the_nearest_rank_95th_percentile() {
        fn ms(values: &[u64]) -> Vec<Duration> {
            values.iter().map(|&v| Duration::from_millis(v)).collect()
        }
        let twenty: Vec<u64> = (1..=20).collect();
        let summary = Summary::of(&ms(&twenty));
        assert_eq!(summary.samples, 20);
        assert!((summary.mean_ms - 10.5).abs() < 1e-9);
        // The 19th of 20, sorted.
        assert!((summary.p95_ms - 19.0).abs() < 1e-9);
        let one = Summary::of(&ms(&[7]));
        assert!((one.mean_ms - 7.0).abs() < 1e-9 && (one.p95_ms - 7.0).abs() < 1e-9);
        assert!(Summary::of(&[]).mean_ms.is_nan());
    }

    #[test]
    fn the_pipelined_copy_returns_each_frame_in_order_with_its_own_bytes() {
        let (w, h) = (64, 48);
        let mut c = software_compositor(w, h).unwrap();
        let frames: Vec<Frame> = (0..7u8)
            .map(|i| {
                let picture: Vec<u8> = (0..w * h)
                    .flat_map(|p| [i * 30, (p % 251) as u8 / 2, 9, 200])
                    .collect();
                let key = painted_key(w, h, &picture);
                c.upload_painted(key, w, h, &picture).unwrap();
                Frame::new([0, 0, 0, 255], vec![Layer::picture(key).with_blur(1.5)])
            })
            .collect();
        let expected: Vec<Vec<u8>> = frames.iter().map(|f| c.draw(f).unwrap()).collect();
        for depth in [2, 3] {
            let mut pipeline = Pipeline::new(&mut c, depth);
            let mut out = Vec::new();
            for (n, frame) in frames.iter().enumerate() {
                out.extend(pipeline.push(frame).unwrap());
                // Nothing comes out until the ring is full; then one frame for each pushed.
                assert_eq!(out.len(), (n + 1).saturating_sub(depth));
            }
            while let Some(rgba) = pipeline.pop().unwrap() {
                out.push(rgba);
            }
            assert_eq!(out.len(), frames.len());
            assert!(
                out == expected,
                "depth {depth}: a frame came out wrong or out of order"
            );
        }
        // A refused frame leaves the ring as it was.
        let mut pipeline = Pipeline::new(&mut c, 2);
        let unknown = Frame::new([0; 4], vec![Layer::picture([7; 16])]);
        assert!(pipeline.push(&unknown).is_err());
        assert!(pipeline.push(&frames[0]).unwrap().is_none());
        assert!(pipeline.push(&frames[1]).unwrap().is_none());
        assert!(pipeline.pop().unwrap().unwrap() == expected[0]);
    }

    #[test]
    fn on_a_software_adapter_every_stack_is_timed_at_480_x_270() {
        let mut open = |w, h| software_compositor(w, h);
        let options = Options {
            frames: 3,
            warmup: 2,
            ring: 2,
            uploads: 2,
        };
        let report = run(&mut open, &[(480, 270)], &options).unwrap();
        assert!(report.software);
        let names: Vec<&str> = report.stacks.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["reference-480x270", "heavy-480x270"]);
        for s in &report.stacks {
            for (summary, samples) in [
                (s.one_frame, 3),
                (s.pipelined, 3),
                (s.upload, 2),
                (s.upload_and_frame, 2),
            ] {
                assert_eq!(summary.samples, samples, "{}", s.name);
                assert!(summary.mean_ms > 0.0 && summary.p95_ms >= 0.0, "{}", s.name);
            }
            assert!(s.upload_and_frame.mean_ms >= s.upload.mean_ms);
            let phases: Vec<&str> = s.phases.iter().map(|p| p.name).collect();
            assert_eq!(phases, ["one_frame", "pipelined", "new_upload"]);
        }
        let parsed: serde_json::Value = serde_json::from_str(&to_json(&report)).unwrap();
        assert_eq!(parsed["stacks"][1]["pipelined"]["samples"], 3);
        assert_eq!(parsed["options"]["ring"], 2);
        assert_eq!(lines(&report).len(), 2);
        assert!(run(&mut open, &[(480, 270)], &Options { ring: 1, ..options }).is_err());
    }
}
