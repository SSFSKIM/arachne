//! Streams built through the core's probe kit (`arachne_core::probes`), written as XF bytes into a
//! temporary directory: the one test module beside `tests/contract*.rs` that may name the core
//! beyond its XF schema (`ci/oracle_check.py`). Streams are never stored in the repository; the
//! tests that judge them read the bytes back through the XF reader alone.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

/// What a probe asserts at a pixel, in the oracle's terms.
#[derive(Clone, Debug, PartialEq)]
pub enum Expect {
    /// The core draws these bytes there.
    Bytes([u8; 4]),
    /// The record in this slot is hidden there.
    Hidden(u32),
}

/// One check: the frame's delta `seq`, the pixel, and what is asserted there.
#[derive(Clone, Debug)]
pub struct Check {
    pub seq: u64,
    pub x: u32,
    pub y: u32,
    pub expect: Expect,
    pub what: &'static str,
}

/// A probe's stream, as a file, and its checks.
#[derive(Clone, Debug)]
pub struct ProbeFile {
    pub name: &'static str,
    pub path: PathBuf,
    pub checks: Vec<Check>,
}

/// The probe kit's probes, each written to `dir` as an XF file.
pub fn probe_files(dir: &Path) -> Vec<ProbeFile> {
    std::fs::create_dir_all(dir).unwrap();
    arachne_core::probes::all()
        .into_iter()
        .map(|p| {
            let path = dir.join(format!("{}.xf", p.name));
            std::fs::write(&path, arachne_core::xf::wire::encode_file(&p.deltas)).unwrap();
            let checks = p
                .checks
                .iter()
                .map(|c| Check {
                    seq: p.deltas[c.frame].seq,
                    x: c.x,
                    y: c.y,
                    expect: match c.core {
                        arachne_core::probes::Expect::Bytes(b) => Expect::Bytes(b),
                        arachne_core::probes::Expect::Hidden(s) => Expect::Hidden(s),
                    },
                    what: c.what,
                })
                .collect();
            ProbeFile {
                name: p.name,
                path,
                checks,
            }
        })
        .collect()
}

/// A fresh scratch directory under the build's temporary directory.
pub fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}
