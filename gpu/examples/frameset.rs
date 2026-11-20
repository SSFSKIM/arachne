//! Draws the frame set on this machine's GPU (the ordinary door: never a software adapter) and
//! writes each frame as raw RGBA with `manifest.json`, the frames' digests and the device's pin.
//!
//!     cargo run --release --locked --features bench --example frameset -- --out <dir>
//!         [--size <width>x<height>] [--digests-only]
//!
//! `--size` draws every frame at one size instead of the set's own; `--digests-only` writes the
//! manifest alone.

use std::path::PathBuf;

use arachne_gpu::Compositor;
use arachne_gpu::bench::frameset::{self, Sizes};

const USAGE: &str = "usage: frameset --out <dir> [--size <width>x<height>] [--digests-only]";

fn size(value: &str) -> Result<(u32, u32), String> {
    let (w, h) = value
        .split_once('x')
        .ok_or_else(|| format!("{value}: not <width>x<height>"))?;
    let parse = |v: &str| v.parse::<u32>().map_err(|e| format!("{value}: {e}"));
    Ok((parse(w)?, parse(h)?))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut out = None;
    let mut sizes = Sizes::Set;
    let mut frames = true;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => out = Some(PathBuf::from(args.next().ok_or(USAGE)?)),
            "--size" => {
                let (w, h) = size(&args.next().ok_or(USAGE)?)?;
                sizes = Sizes::All(w, h);
            }
            "--digests-only" => frames = false,
            _ => return Err(format!("{arg}: {USAGE}").into()),
        }
    }
    let out = out.ok_or(USAGE)?;
    let mut open = |w, h| Compositor::new(w, h);
    let manifest = frameset::draw(&mut open, sizes, &out, frames)?;
    println!("ADAPTER: {}", manifest.adapter);
    for frame in &manifest.frames {
        println!("{} {}", frame.sha256, frame.name);
    }
    println!(
        "frameset: {} frames in {}",
        manifest.frames.len(),
        out.join("manifest.json").display()
    );
    Ok(())
}
