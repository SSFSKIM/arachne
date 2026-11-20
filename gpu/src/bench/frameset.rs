//! The frame set: the reference and heavy stacks at 1920 x 1080 and 3840 x 2160 and one frame per
//! effect probe at 1920 x 1080 ([`super::stacks`]), each drawn, read back and written as raw RGBA
//! with a manifest of their digests. A device is compared with another frame by frame, by name,
//! within one version of the set.

use std::path::Path;

use super::stacks::{self, PROBE_SIZE, PROBES, REFERENCE_SIZES, Stack};
use super::{Error, Open, hex_digest, json};
use crate::{AdapterSummary, Compositor, Pin};

/// What the frames are drawn at: the set's own sizes, or every frame at one size (a quick run,
/// such as CI's on a software adapter).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sizes {
    Set,
    All(u32, u32),
}

/// One frame of the set as drawn.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Drawn {
    pub name: String,
    pub width: u32,
    pub height: u32,
    /// The SHA-256 of the frame's raw RGBA rows, top first, painted: its file's bytes.
    pub sha256: String,
}

/// The set as one device drew it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Manifest {
    pub pin: Pin,
    pub adapter: AdapterSummary,
    pub software: bool,
    pub frames: Vec<Drawn>,
}

/// Which stack to build, at which size.
enum Entry {
    Reference,
    Heavy,
    Probe(&'static str),
}

/// The set's frames, grouped by size, in order.
fn plan(sizes: Sizes) -> Vec<((u32, u32), Vec<Entry>)> {
    let all = || {
        let mut entries = vec![Entry::Reference, Entry::Heavy];
        entries.extend(PROBES.iter().copied().map(Entry::Probe));
        entries
    };
    match sizes {
        Sizes::All(width, height) => vec![((width, height), all())],
        Sizes::Set => REFERENCE_SIZES
            .iter()
            .map(|&size| {
                if size == PROBE_SIZE {
                    (size, all())
                } else {
                    (size, vec![Entry::Reference, Entry::Heavy])
                }
            })
            .collect(),
    }
}

fn build(entry: &Entry, width: u32, height: u32) -> Stack {
    match entry {
        Entry::Reference => stacks::reference(width, height, 0),
        Entry::Heavy => stacks::heavy(width, height, 0),
        Entry::Probe(name) => stacks::probe(name, width, height),
    }
}

/// Draws the set, one compositor per size, each opened by `open`. Writes `manifest.json` into
/// `out`, and with `frames`, each frame as `<name>.rgba` beside it. Fails if two compositors are
/// on different devices.
pub fn draw(
    open: &mut Open<'_>,
    sizes: Sizes,
    out: &Path,
    frames: bool,
) -> Result<Manifest, Error> {
    std::fs::create_dir_all(out)?;
    let mut device: Option<(Pin, AdapterSummary, bool)> = None;
    let mut drawn = Vec::new();
    for ((width, height), entries) in plan(sizes) {
        let mut compositor: Compositor = open(width, height)?;
        let this = (
            compositor.pin(),
            compositor.adapter(),
            compositor.software(),
        );
        if let Some(first) = device.as_ref().filter(|first| **first != this) {
            return Err(format!("the set was drawn on two devices: {first:?} and {this:?}").into());
        }
        if device.is_none() {
            device = Some(this);
        }
        for entry in &entries {
            let stack = build(entry, width, height);
            stack.upload(&mut compositor)?;
            let rgba = compositor.draw(&stack.frame)?;
            stack.release(&mut compositor);
            if frames {
                std::fs::write(out.join(format!("{}.rgba", stack.name)), &rgba)?;
            }
            drawn.push(Drawn {
                name: stack.name,
                width,
                height,
                sha256: hex_digest(&rgba),
            });
        }
    }
    let (pin, adapter, software) = device.ok_or("the set has no frame")?;
    let manifest = Manifest {
        pin,
        adapter,
        software,
        frames: drawn,
    };
    std::fs::write(out.join("manifest.json"), to_json(&manifest, frames))?;
    Ok(manifest)
}

/// The manifest as JSON: the set's version, the device, and each frame's name, size, file and
/// digest.
pub fn to_json(manifest: &Manifest, frames: bool) -> String {
    let entries: Vec<String> = manifest
        .frames
        .iter()
        .map(|f| {
            let file = if frames {
                json::string(&format!("{}.rgba", f.name))
            } else {
                "null".into()
            };
            json::object(&[
                ("name", json::string(&f.name)),
                ("width", f.width.to_string()),
                ("height", f.height.to_string()),
                ("file", file),
                ("sha256", json::string(&f.sha256)),
            ])
        })
        .collect();
    let top = json::object(&[
        ("frame_set", stacks::VERSION.to_string()),
        (
            "format",
            json::string(
                "raw RGBA8 rows, top first, painted (colour multiplied by alpha); sha256 over \
                 those bytes",
            ),
        ),
        ("pin", json::pin(&manifest.pin)),
        ("adapter", json::adapter(&manifest.adapter)),
        ("software", manifest.software.to_string()),
        ("frames", json::array(&entries)),
    ]);
    format!("{top}\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::software_compositor;

    fn temporary(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "arachne-gpu-frameset-{name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn the_set_draws_at_its_own_sizes_in_order() {
        let names: Vec<(u32, usize)> = plan(Sizes::Set)
            .iter()
            .map(|((width, _), entries)| (*width, entries.len()))
            .collect();
        assert_eq!(names, [(1920, 17), (3840, 2)]);
        assert_eq!(plan(Sizes::All(480, 270))[0].1.len(), 17);
    }

    #[test]
    fn on_a_software_adapter_the_set_is_drawn_written_and_repeated_at_480_x_270() {
        let mut open = |w, h| software_compositor(w, h);
        let (first, second) = (temporary("first"), temporary("second"));
        let a = draw(&mut open, Sizes::All(480, 270), &first, true).unwrap();
        let b = draw(&mut open, Sizes::All(480, 270), &second, false).unwrap();
        assert!(a.software);
        assert_eq!(a.adapter.device_type, "Cpu");
        assert_eq!(a.frames.len(), 2 + PROBES.len());
        // The same set, drawn again, is the same bytes.
        assert_eq!(a.frames, b.frames);
        // A depth off the focus resolves to the blur it declares, exactly; every other frame
        // draws something of its own (a window in a window at the frame's corners may draw what
        // one window does, so it is left out).
        let digest = |name: &str| {
            let name = format!("probe-{name}-480x270");
            a.frames
                .iter()
                .find(|f| f.name == name)
                .unwrap()
                .sha256
                .clone()
        };
        assert_eq!(digest("focus-off"), digest("blur-step-sigma-6"));
        let others: std::collections::HashSet<&str> = a
            .frames
            .iter()
            .filter(|f| !f.name.contains("focus-off") && !f.name.contains("window-nested"))
            .map(|f| f.sha256.as_str())
            .collect();
        assert_eq!(others.len(), a.frames.len() - 2);
        for frame in &a.frames {
            let bytes = std::fs::read(first.join(format!("{}.rgba", frame.name))).unwrap();
            assert_eq!(bytes.len(), 480 * 270 * 4);
            assert_eq!(hex_digest(&bytes), frame.sha256);
            assert!(!second.join(format!("{}.rgba", frame.name)).exists());
        }
        let manifest: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(first.join("manifest.json")).unwrap())
                .unwrap();
        assert_eq!(manifest["frame_set"], stacks::VERSION);
        assert_eq!(manifest["pin"]["wgpu"], crate::WGPU_VERSION);
        assert_eq!(manifest["frames"][0]["name"], "reference-480x270");
        assert_eq!(manifest["frames"][0]["sha256"], a.frames[0].sha256.as_str());
        for dir in [first, second] {
            std::fs::remove_dir_all(dir).unwrap();
        }
    }
}
