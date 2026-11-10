//! `ORACLE_FORMAT` 1 round-trips: the oracle's own reader and `read.py` read back exactly what it
//! wrote, and the command line writes a file `read.py` reads.

mod common;

use std::path::{Path, PathBuf};
use std::process::Command;

use arachne_oracle::out::{
    BandRow, ColourRow, CoverRow, Decisions, Fragment, Header, PairRow, PixelColour, Refused,
    STATUS_UNABLE, canonical, read, write,
};
use common::*;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("arachne-oracle-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// What `read.py --canonical` prints for `file`.
fn read_py(file: &Path) -> String {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("read.py");
    let out = Command::new("python3")
        .arg("-I")
        .arg(script)
        .arg("--canonical")
        .arg(file)
        .output()
        .expect("python3 runs");
    assert!(
        out.status.success(),
        "read.py failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

fn sample() -> Decisions {
    let row = |x: u32, y: u32, fragment: u32| CoverRow {
        x,
        y,
        fragment,
        cover: std::array::from_fn(|s| (s % 4) as u8),
        cell_lo: std::array::from_fn(|s| s as u8),
        cell_hi: std::array::from_fn(|s| 240 + s as u8),
        pixel_lo: 3,
        pixel_hi: 250,
        pixel_area: 0.1 + 0.2,
        cell_area: std::array::from_fn(|s| match s {
            0 => -0.0,
            1 => f64::NAN,
            2 => 1e-300,
            3 => 5e-324,
            _ => s as f64 / 7.0,
        }),
        depth: [f64::NEG_INFINITY; 16],
        band: std::array::from_fn(|s| s as f64 * 0.125),
    };
    Decisions {
        header: Header {
            xf: (1, 4),
            width: 5,
            height: 3,
            window: [1, 0, 4, 3],
            status: 0,
            seq: 1 << 40,
            offsets: arachne_oracle::cover::SAMPLE_OFFSETS,
        },
        fragments: vec![
            Fragment {
                id: 0,
                entry: 2,
                slot: 9,
                object: 77,
                part: 3,
                sub: 5,
                layer: 1,
                kind: 2,
                flags: 1,
                group: 4,
                plane: u32::MAX,
                epsilon: f64::INFINITY,
                bbox: [1, 0, 3, 2],
            },
            Fragment {
                id: 1,
                entry: 0,
                slot: 0,
                object: 1,
                part: 0,
                sub: 123_456,
                layer: 3,
                kind: 6,
                flags: 0,
                group: u32::MAX,
                plane: 7,
                epsilon: 1.0 / 256.0,
                bbox: [-2, -1, 4, 3],
            },
        ],
        covers: vec![row(1, 0, 0), row(2, 0, 0), row(2, 0, 1), row(3, 2, 1)],
        bands: vec![
            BandRow {
                x: 2,
                y: 0,
                fragment: 1,
                which: 0,
                o: std::array::from_fn(|s| -1.0 - s as f64 / 64.0),
                r: [1.0 / 1_099_511_627_776.0; 16],
                s_minus: [0.25; 16],
                s_plus: std::array::from_fn(|s| if s == 3 { f64::NAN } else { 0.5 }),
                status: std::array::from_fn(|s| (s % 4) as u8),
            },
            BandRow {
                x: 2,
                y: 0,
                fragment: 1,
                which: 1,
                o: [f64::NEG_INFINITY; 16],
                r: [0.0; 16],
                s_minus: [0.0; 16],
                s_plus: [0.0; 16],
                status: [1; 16],
            },
        ],
        pairs: vec![
            PairRow {
                x: 2,
                y: 0,
                a: 0,
                b: 1,
                sample: 0,
                edges: 1,
                label: 5,
            },
            PairRow {
                x: 2,
                y: 0,
                a: 0,
                b: 1,
                sample: 15,
                edges: 7,
                label: 0,
            },
        ],
        colours: vec![
            ColourRow {
                fragment: 0,
                kind: 1,
                rgba: [255, 128, 0, 255],
            },
            ColourRow {
                fragment: 1,
                kind: 2,
                rgba: [0; 4],
            },
        ],
        pixel_colours: vec![PixelColour {
            x: 2,
            y: 0,
            fragment: 1,
            lo: [1, 2, 3, 4],
            hi: [250, 251, 252, 253],
        }],
        refused: vec![
            Refused {
                reason: 8,
                kind: 1,
                cause: 0,
                entry: 4,
                slot: 5,
                object: 6,
                bits: 1 << 63,
                at_seq: u64::MAX,
                detail: "a clip — refused".into(),
            },
            // The stream-wide refusal: the seq of the delta that refused the stream, and its code.
            Refused {
                reason: arachne_oracle::stream::Reason::StreamRefused.code(),
                kind: 0xff,
                cause: arachne_oracle::stream::StreamCause::SlotNotHeld as u8,
                entry: u32::MAX,
                slot: 7,
                object: u32::MAX,
                bits: 0,
                at_seq: (1 << 40) - 3,
                detail: "a held row for slot 7, which the stream does not hold".into(),
            },
        ],
    }
}

#[test]
fn the_file_round_trips_through_both_readers() {
    let d = sample();
    let file = scratch("roundtrip").join("d.oracle");
    std::fs::write(&file, write(&d)).unwrap();
    let back = read(&std::fs::read(&file).unwrap()).unwrap();
    assert_eq!(canonical(&back), canonical(&d));
    assert_eq!(read_py(&file), canonical(&d));
}

#[test]
fn every_section_starts_where_numpy_can_read_it_and_the_pixel_index_counts_rows() {
    let b = write(&sample());
    let n = u32::from_le_bytes(b[80..84].try_into().unwrap()) as usize;
    let mut tags = Vec::new();
    for i in 0..n {
        let at = 88 + 24 * i;
        tags.push(String::from_utf8(b[at..at + 4].to_vec()).unwrap());
        let off = u64::from_le_bytes(b[at + 8..at + 16].try_into().unwrap());
        assert_eq!(off % 8, 0);
    }
    assert_eq!(
        tags,
        [
            "FRAG", "COVR", "PIXL", "BAND", "PAIR", "COLR", "COLP", "REFU", "TEXT"
        ]
    );
    // The window is 3 x 3 pixels; pixel (2, 0) holds rows 1 and 2.
    let pixl = 88 + 24 * 2;
    let off = u64::from_le_bytes(b[pixl + 8..pixl + 16].try_into().unwrap()) as usize;
    let count = u64::from_le_bytes(b[pixl + 16..pixl + 24].try_into().unwrap());
    assert_eq!(count, 9);
    let at = off + 12;
    let first = u64::from_le_bytes(b[at..at + 8].try_into().unwrap());
    let rows = u32::from_le_bytes(b[at + 8..at + 12].try_into().unwrap());
    assert_eq!((first, rows), (1, 2));
}

#[test]
fn the_command_line_decides_a_frame_read_from_an_xf_file() {
    let v = view(16, 16);
    let s = path_shape(polygon(&[
        frame(&v, [2.0, 2.0]),
        frame(&v, [6.5, 2.0]),
        frame(&v, [6.5, 6.0]),
        frame(&v, [2.0, 6.0]),
    ]));
    let r = path(0, s.key(), true, None);
    let deltas = vec![delta(4, Some(v), vec![s], vec![r])];
    let dir = scratch("cli");
    let xf = dir.join("stream.xf");
    arachne_core::xf::wire::write_file(&xf, &deltas).unwrap();
    let out = dir.join("decisions.oracle");
    let run = Command::new(env!("CARGO_BIN_EXE_arachne-oracle"))
        .args(["decide", "--xf"])
        .arg(&xf)
        .args(["--seq", "4", "--out"])
        .arg(&out)
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert!(String::from_utf8_lossy(&run.stdout).starts_with("decide: seq 4 16x16"));
    let text = read_py(&out);
    // The writer stamps each delta with the lowest version its content needs: decide what it
    // wrote.
    let written = arachne_core::xf::wire::read_file(&xf).unwrap();
    let want = canonical(&arachne_oracle::decide::decide_at(&written, 4, None).unwrap());
    assert_eq!(text, want);
    // The square covers pixels 2..6 by 2..6 whole and column 6 by half.
    assert!(text.contains("cover x=2 y=2 fragment=0 cover=1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1"));
    let half = arachne_oracle::out::read(&std::fs::read(&out).unwrap())
        .unwrap()
        .covers
        .into_iter()
        .find(|c| (c.x, c.y) == (6, 3))
        .unwrap();
    assert!((half.pixel_area - 0.5).abs() < 1e-12, "{}", half.pixel_area);
    // A seq the file lacks is a usage error, exit 2.
    let missing = Command::new(env!("CARGO_BIN_EXE_arachne-oracle"))
        .args(["decide", "--xf"])
        .arg(&xf)
        .args(["--seq", "5", "--out"])
        .arg(&out)
        .output()
        .unwrap();
    assert_eq!(missing.status.code(), Some(2));
}

#[test]
fn a_refused_frame_writes_its_refusal() {
    let v = view(8, 8);
    let mut r = arachne_core::xf::Record::new(arachne_core::xf::RecordKind::Emission, 3, 30);
    r.body = arachne_core::xf::Body::Opaque(vec![]);
    let d =
        arachne_oracle::decide::decide_at(&[delta(1, Some(v), vec![], vec![r])], 1, None).unwrap();
    assert_eq!(d.header.status, STATUS_UNABLE);
    let file = scratch("refused").join("d.oracle");
    std::fs::write(&file, write(&d)).unwrap();
    let text = read_py(&file);
    assert!(text.contains("status=truth_unable"), "{text}");
    assert!(
        text.contains("refused reason=2 kind=66 cause=0 entry=0 slot=3 object=30"),
        "{text}"
    );
}
