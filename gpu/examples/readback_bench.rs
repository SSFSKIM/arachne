//! Times the reference and heavy stacks on this machine's GPU (the ordinary door: never a
//! software adapter), with the one-frame copy and with the pipelined copy, and one frame after a
//! new upload of every picture; writes the report as JSON and a line per stack.
//!
//!     cargo run --release --locked --features bench --example readback_bench -- --out <file>
//!         [--frames 300] [--warmup 30] [--ring 3] [--uploads 30] [--size <width>x<height>]...
//!
//! With no `--size`, the stacks are timed at 1920 x 1080 and 3840 x 2160.

use std::path::PathBuf;

use arachne_gpu::Compositor;
use arachne_gpu::bench::stacks::REFERENCE_SIZES;
use arachne_gpu::bench::timing::{self, Options};

const USAGE: &str = "usage: readback_bench --out <file> [--frames N] [--warmup N] [--ring N] \
                     [--uploads N] [--size <width>x<height>]...";

fn size(value: &str) -> Result<(u32, u32), String> {
    let (w, h) = value
        .split_once('x')
        .ok_or_else(|| format!("{value}: not <width>x<height>"))?;
    let parse = |v: &str| v.parse::<u32>().map_err(|e| format!("{value}: {e}"));
    Ok((parse(w)?, parse(h)?))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut out = None;
    let mut options = Options::default();
    let mut sizes = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or(USAGE);
        match arg.as_str() {
            "--out" => out = Some(PathBuf::from(value()?)),
            "--frames" => options.frames = value()?.parse()?,
            "--warmup" => options.warmup = value()?.parse()?,
            "--ring" => options.ring = value()?.parse()?,
            "--uploads" => options.uploads = value()?.parse()?,
            "--size" => sizes.push(size(&value()?)?),
            _ => return Err(format!("{arg}: {USAGE}").into()),
        }
    }
    let out = out.ok_or(USAGE)?;
    if sizes.is_empty() {
        sizes.extend(REFERENCE_SIZES);
    }
    let mut open = |w, h| Compositor::new(w, h);
    let report = timing::run(&mut open, &sizes, &options)?;
    timing::write(&report, &out)?;
    println!("ADAPTER: {}", report.adapter);
    for line in timing::lines(&report) {
        println!("{line}");
    }
    println!("readback_bench: {}", out.display());
    Ok(())
}
