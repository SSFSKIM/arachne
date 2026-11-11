//! `arachne-oracle decide --xf <file> --seq <delta seq> --out <file> [--pixels x0,y0,x1,y1]`:
//! the oracle's decisions for the frame drawn at one delta of an XF file, written as
//! `ORACLE_FORMAT` 1 (`FORMAT.md`). Exit 0 when the decisions are written (a refused frame
//! included: its file says `truth_unable` and why), 2 on a usage or input error.

use std::path::PathBuf;
use std::process::ExitCode;

use arachne_oracle::decide::decide_at_wire;
use arachne_oracle::out::{STATUS_DECIDED, write};
use arachne_oracle::stream::Reason;

const USAGE: &str = "usage: arachne-oracle decide --xf <file> --seq <delta seq> --out <file> [--pixels x0,y0,x1,y1]";

struct Args {
    xf: PathBuf,
    seq: u64,
    out: PathBuf,
    pixels: Option<[u32; 4]>,
}

fn parse(mut args: impl Iterator<Item = String>) -> Result<Args, String> {
    if args.next().as_deref() != Some("decide") {
        return Err(USAGE.into());
    }
    let (mut xf, mut seq, mut out, mut pixels) = (None, None, None, None);
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("{flag} needs a value\n{USAGE}"))?;
        match flag.as_str() {
            "--xf" => xf = Some(PathBuf::from(value)),
            "--seq" => seq = Some(value.parse::<u64>().map_err(|e| format!("--seq: {e}"))?),
            "--out" => out = Some(PathBuf::from(value)),
            "--pixels" => {
                let v: Vec<u32> = value
                    .split(',')
                    .map(|s| s.trim().parse::<u32>())
                    .collect::<Result<_, _>>()
                    .map_err(|e| format!("--pixels: {e}"))?;
                let [x0, y0, x1, y1] = v[..] else {
                    return Err("--pixels takes x0,y0,x1,y1".into());
                };
                pixels = Some([x0, y0, x1, y1]);
            }
            other => return Err(format!("unknown flag {other}\n{USAGE}")),
        }
    }
    Ok(Args {
        xf: xf.ok_or(USAGE)?,
        seq: seq.ok_or(USAGE)?,
        out: out.ok_or(USAGE)?,
        pixels,
    })
}

fn run() -> Result<(), String> {
    let args = parse(std::env::args().skip(1))?;
    let (_, deltas) = arachne_oracle::source::read_file(&args.xf)
        .map_err(|e| format!("{}: {}", args.xf.display(), e.0))?;
    let d = decide_at_wire(&deltas, args.seq, args.pixels).map_err(|e| e.to_string())?;
    std::fs::write(&args.out, write(&d)).map_err(|e| format!("{}: {e}", args.out.display()))?;
    let h = &d.header;
    if h.status == STATUS_DECIDED {
        println!(
            "decide: seq {} {}x{} window {:?}: {} fragments, {} rows",
            h.seq,
            h.width,
            h.height,
            h.window,
            d.fragments.len(),
            d.covers.len()
        );
    } else {
        let reasons: Vec<String> = d
            .refused
            .iter()
            .map(|r| {
                let name = Reason::from_code(r.reason).map_or("unknown", Reason::name);
                format!("{name}: {}", r.detail)
            })
            .collect();
        println!(
            "decide: seq {} truth_unable ({})",
            h.seq,
            reasons.join("; ")
        );
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("arachne-oracle: {e}");
            ExitCode::from(2)
        }
    }
}
