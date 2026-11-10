//! The contract's stroke behaviour fixtures (§12, "Stroke behaviour fixtures (§8)"):
//! `join_occurrences`, `join_closing` and `guarded_seam`, each a scene whose expected coverage
//! follows from §8 alone.
//!
//! A fixture directory under `tests/fixtures/` holds `expected.json`: the verdict (agreement), the
//! stroke fragment it reads, and per pixel the samples whose cell bytes §8 fixes, with the bytes the
//! named wrong reading would give instead. The scene, as §12 gives it, is built here and never
//! committed as a stream: the directory pins it by `scene.sha256`, the sha256 of its XF file's
//! bytes, and the tests decide on the scene read back from those bytes. A test holds both files to
//! what is stated here, and `ORACLE_BLESS_FIXTURES=1` rewrites them from it.

mod common;

use std::path::{Path, PathBuf};

use arachne_core::xf::{
    Background, Camera, Dash, Delta, Joint, LightModel, Record, Shape, View, wire,
};
use arachne_oracle::cover::sample_cell;
use arachne_oracle::decide::decide_at;
use arachne_oracle::out::{Cover, CoverRow, Decisions};
use common::{delta, path, path_shape, sha256_hex};

/// The contract's view F: 16 x 16 px over a 16 x 16 unit frame about the origin, `xx = 1`,
/// `yy = −1`, `x0 = y0 = 8`, an input grid of `2^-20`, a black background.
fn view_f() -> View {
    View {
        pixel_width: 16,
        pixel_height: 16,
        frame_width: 16.0,
        frame_height: 16.0,
        frame_center: [0.0; 3],
        mapping_centre: [0.0; 3],
        background: Background::Color([0.0, 0.0, 0.0, 1.0]),
        camera: Camera::Flat,
        cairo_line_width_multiple: 1.0,
        use_z_index: false,
        light: LightModel::Ce,
        input_grid: 1.0 / 1_048_576.0,
    }
}

/// A line from `p` to `q` as §12 writes it: one cubic with controls the nearest binary64 values to
/// `(2p + q) / 3` and `(p + 2q) / 3`.
fn line(p: [f64; 2], q: [f64; 2]) -> [[f64; 3]; 4] {
    let third = |a: f64, b: f64| (2.0 * a + b) / 3.0;
    [
        [p[0], p[1], 0.0],
        [third(p[0], q[0]), third(p[1], q[1]), 0.0],
        [third(q[0], p[0]), third(q[1], p[1]), 0.0],
        [q[0], q[1], 0.0],
    ]
}

fn lines(points: &[[f64; 2]]) -> Vec<[f64; 3]> {
    points.windows(2).flat_map(|w| line(w[0], w[1])).collect()
}

/// One path record at slot 0 (object 100), stroked 1 unit wide, no fill.
fn stroke(
    points: Vec<[f64; 3]>,
    joint: Joint,
    joins: Option<Vec<u8>>,
    dash: Option<Dash>,
) -> Vec<Delta> {
    let shape: Shape = path_shape(points);
    let mut r: Record = path(0, shape.key(), false, Some(1.0));
    r.paint.joint = joint;
    r.joins = joins;
    r.dash = dash;
    vec![delta(1, Some(view_f()), vec![shape], vec![r])]
}

/// Where `join_occurrences`'s spikes start, in frame `x`: each spike's first cubic is then 6.1 px
/// long, 0.1 px from a step of its centreline count, clear of `curve_recut` (§6) as §12 requires.
const SPIKE_X: f64 = -7.1;

/// `join_occurrences`'s two spikes, each drawn twice, as four open subpaths of two straight
/// cubics: the upper from `(SPIKE_X, 4)` to `(-1, 4)` and back to `(-7, 5)`, twice; the lower the
/// same 8 units lower, twice.
fn spikes() -> Vec<[f64; 3]> {
    let spike = |y: f64| lines(&[[SPIKE_X, y], [-1.0, y], [-7.0, y + 1.0]]);
    [spike(4.0), spike(4.0), spike(-4.0), spike(-4.0)].concat()
}

/// `join_occurrences`: each spike's corner one key met twice; the upper key's decisions miter then
/// bevel, the lower key's bevel then miter.
fn occurrences(joint: Joint, joins: Option<Vec<u8>>) -> Vec<Delta> {
    stroke(spikes(), joint, joins, None)
}
const OCCURRENCE_BYTES: [u8; 8] = [0, 0b11, 0, 0b01, 0, 0b01, 0, 0b11];

const SQUARE: [[f64; 2]; 5] = [
    [-4.0, -4.0],
    [4.0, -4.0],
    [4.0, 4.0],
    [-4.0, 4.0],
    [-4.0, -4.0],
];

/// `join_closing`: a closed square.
fn closing(joint: Joint, joins: Option<Vec<u8>>) -> Vec<Delta> {
    stroke(lines(&SQUARE), joint, joins, None)
}
const CLOSING_BYTES: [u8; 4] = [0b01, 0b11, 0b11, 0b11];

/// `guarded_seam`: a closed subpath out to `x = 100000` and back, dashed.
fn seam() -> Vec<Delta> {
    let dash = Dash {
        period: 2.0,
        duty: 0.75,
        phase: 0.25,
    };
    stroke(lines(&FAR), Joint::Miter, None, Some(dash))
}

/// The closed subpath of `guarded_seam`, from the frame out to `x = 100000` and back.
const FAR: [[f64; 2]; 6] = [
    [-4.0, -4.0],
    [4.0, -4.0],
    [100000.0, -4.0],
    [100000.0, 4.0],
    [-4.0, 4.0],
    [-4.0, -4.0],
];

/// Samples whose cell bytes a fixture fixes at one pixel of the stroke layer.
struct Check {
    pixel: [u32; 2],
    samples: &'static [usize],
    cell_bytes: &'static [u8],
    wrong_reading: &'static [u8],
    what: &'static str,
}

struct Fixture {
    name: &'static str,
    rules: &'static [&'static str],
    scene: Vec<Delta>,
    wrong_reading: &'static str,
    checks: Vec<Check>,
}

/// The fragment every fixture reads: object 100's stroke (`layer` 2, `sub` 2).
const OBJECT: u32 = 100;
const STROKE: u8 = 2;

fn fixtures() -> Vec<Fixture> {
    vec![
        Fixture {
            name: "join_occurrences",
            rules: &["join_mask@1"],
            scene: occurrences(Joint::Miter, Some(OCCURRENCE_BYTES.to_vec())),
            wrong_reading: "one decision per key applied to both its occurrences: the first \
                            bevels the lower spike, the last the upper",
            checks: vec![
                Check {
                    pixel: [8, 4],
                    samples: &[0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15],
                    cell_bytes: &[255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0],
                    wrong_reading: &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    what: "the upper spike's tip, drawn by its first occurrence's miter, over \
                           frame y above about 3.52; uncovered under each key's last decision",
                },
                Check {
                    pixel: [8, 12],
                    samples: &[0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15],
                    cell_bytes: &[255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0],
                    wrong_reading: &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    what: "the lower spike's tip, drawn by its second occurrence's miter; \
                           uncovered under each key's first decision",
                },
            ],
        },
        Fixture {
            name: "join_closing",
            rules: &["join_mask@1"],
            scene: closing(Joint::Miter, Some(CLOSING_BYTES.to_vec())),
            wrong_reading: "the closing join read from the last cubic's byte (a miter)",
            checks: vec![
                Check {
                    pixel: [3, 12],
                    samples: &[2, 3, 6, 7],
                    cell_bytes: &[128, 255, 0, 128],
                    wrong_reading: &[255, 255, 255, 255],
                    what: "the outer corner square of (-4, -4), cut by the closing join's bevel",
                },
                Check {
                    pixel: [12, 12],
                    samples: &[5],
                    cell_bytes: &[255],
                    wrong_reading: &[255],
                    what: "the outermost cell of (4, -4)'s mitred corner",
                },
                Check {
                    pixel: [12, 3],
                    samples: &[9],
                    cell_bytes: &[255],
                    wrong_reading: &[255],
                    what: "the outermost cell of (4, 4)'s mitred corner",
                },
                Check {
                    pixel: [3, 3],
                    samples: &[10],
                    cell_bytes: &[255],
                    wrong_reading: &[255],
                    what: "the outermost cell of (-4, 4)'s mitred corner",
                },
            ],
        },
        Fixture {
            name: "guarded_seam",
            rules: &["guard_chords@1", "dash_start_least_length@1"],
            scene: seam(),
            wrong_reading: "the runs dashed apart, leaving two butt caps at (-4, -4)",
            checks: vec![Check {
                pixel: [3, 12],
                samples: &[2, 3, 6, 7],
                cell_bytes: &[255, 255, 255, 255],
                wrong_reading: &[0, 0, 0, 0],
                what: "the outer corner square of (-4, -4), stroked as a mitred join across the seam",
            }],
        },
    ]
}

fn json_list<T: std::fmt::Display>(v: &[T]) -> String {
    let items: Vec<String> = v.iter().map(|x| x.to_string()).collect();
    format!("[{}]", items.join(", "))
}

fn expected_json(f: &Fixture) -> String {
    let rules: Vec<String> = f.rules.iter().map(|r| format!("\"{r}\"")).collect();
    let checks: Vec<String> = f
        .checks
        .iter()
        .map(|c| {
            format!(
                "    {{\"pixel\": {}, \"samples\": {}, \"cell_bytes\": {}, \"wrong_reading\": {}, \"what\": \"{}\"}}",
                json_list(&c.pixel),
                json_list(c.samples),
                json_list(c.cell_bytes),
                json_list(c.wrong_reading),
                c.what
            )
        })
        .collect();
    format!(
        "{{\n  \"fixture\": \"{}\",\n  \"kind\": \"stroke_behaviour\",\n  \"rules\": [{}],\n  \
         \"verdict\": \"agreement\",\n  \"seq\": 1,\n  \"fragment\": {{\"object\": {OBJECT}, \"part\": 0, \
         \"layer\": \"stroke\", \"sub\": 2}},\n  \"wrong_reading\": \"{}\",\n  \"checks\": [\n{}\n  ]\n}}\n",
        f.name,
        rules.join(", "),
        f.wrong_reading,
        checks.join(",\n")
    )
}

fn dir(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

#[test]
fn the_fixture_files_pin_the_scenes_and_expectations_stated_here() {
    let bless = std::env::var_os("ORACLE_BLESS_FIXTURES").is_some();
    for f in fixtures() {
        let d = dir(f.name);
        let scene = wire::encode_file(&f.scene);
        let digest = format!("{}\n", sha256_hex(&scene));
        let expected = expected_json(&f);
        if bless {
            std::fs::create_dir_all(&d).unwrap();
            std::fs::write(d.join("scene.sha256"), &digest).unwrap();
            std::fs::write(d.join("expected.json"), &expected).unwrap();
        }
        let pinned = std::fs::read_to_string(d.join("scene.sha256")).expect("scene.sha256");
        assert_eq!(
            pinned, digest,
            "{}: the scene is not the one pinned",
            f.name
        );
        let on_disk = std::fs::read_to_string(d.join("expected.json")).expect("expected.json");
        assert_eq!(on_disk, expected, "{}: expected.json", f.name);
        // The file reads back to deltas that write it again (the writer records the least XF
        // version the scene needs).
        let read = wire::decode_file(&scene).unwrap();
        assert!(wire::encode_file(&read) == scene, "{}: round trip", f.name);
    }
}

/// The decisions on a fixture's scene, read back from its XF file's bytes.
fn decided(name: &str) -> Decisions {
    let f = fixture(name);
    let deltas = wire::decode_file(&wire::encode_file(&f.scene)).expect("the scene reads");
    decide_at(&deltas, 1, None).expect("decided")
}

fn stroke_row(d: &Decisions, [x, y]: [u32; 2]) -> Option<&CoverRow> {
    let f = d
        .fragments
        .iter()
        .find(|f| f.object == OBJECT && f.part == 0 && f.layer == STROKE && f.sub == 2)?;
    d.covers
        .iter()
        .find(|c| c.fragment == f.id && c.x == x && c.y == y)
}

/// The admitted cell bytes `[lo, hi]` and the verdict of the stroke at a sample.
fn admitted(d: &Decisions, pixel: [u32; 2], s: usize) -> (u8, u8, u8) {
    stroke_row(d, pixel).map_or((0, 0, Cover::Absent as u8), |r| {
        (r.cell_lo[s], r.cell_hi[s], r.cover[s])
    })
}

fn agrees(f: &Fixture, d: &Decisions) {
    for c in &f.checks {
        for ((&s, &want), &wrong) in c.samples.iter().zip(c.cell_bytes).zip(c.wrong_reading) {
            let (lo, hi, cover) = admitted(d, c.pixel, s);
            let at = format!(
                "{} pixel {:?} sample {s} (cell {:?})",
                f.name,
                c.pixel,
                sample_cell(s)
            );
            assert!(
                lo <= want && want <= hi,
                "{at}: admits [{lo}, {hi}], not {want}"
            );
            if wrong != want {
                assert!(
                    !(lo <= wrong && wrong <= hi),
                    "{at}: admits the wrong reading's {wrong}"
                );
            }
            let verdict = match want {
                0 => Cover::Absent,
                _ => Cover::Yes,
            };
            assert_eq!(cover, verdict as u8, "{at}");
        }
    }
}

fn fixture(name: &str) -> Fixture {
    fixtures().into_iter().find(|f| f.name == name).unwrap()
}

#[test]
fn join_occurrences_each_occurrence_takes_its_own_decision() {
    let f = fixture("join_occurrences");
    let d = decided(f.name);
    agrees(&f, &d);
    // Each tip covers its pixel over the area of the triangle closed by the bevel's ends, about
    // 0.4736 by §12's coordinates, which it rounds to 5 places.
    for pixel in [[8, 4], [8, 12]] {
        let row = stroke_row(&d, pixel).expect("the tip's pixel");
        assert!(
            (row.pixel_area - 0.4736).abs() < 1e-3,
            "{pixel:?}: area {}",
            row.pixel_area
        );
    }
    // With no joins carried kurbo bevels both corners, so neither pixel is reached: what covers
    // them is the carried decisions alone.
    let bevelled = decide_at(&occurrences(Joint::Miter, None), 1, None).unwrap();
    for pixel in [[8, 4], [8, 12]] {
        let area = stroke_row(&bevelled, pixel).map_or(0.0, |r| r.pixel_area);
        assert_eq!(area, 0.0, "{pixel:?} without joins");
    }
}

#[test]
fn join_closing_takes_the_first_cubics_byte() {
    let f = fixture("join_closing");
    let d = decided(f.name);
    agrees(&f, &d);
    // The closing corner as a bevel strokes it; the other three as kurbo's miter.
    let mitred = decide_at(&closing(Joint::Miter, None), 1, None).unwrap();
    let bevelled = decide_at(&closing(Joint::Bevel, None), 1, None).unwrap();
    let cells = |d: &Decisions, p| stroke_row(d, p).map(|r| (r.cell_lo, r.cell_hi));
    assert_eq!(cells(&d, [3, 12]), cells(&bevelled, [3, 12]));
    for p in [[12, 12], [12, 3], [3, 3]] {
        assert_eq!(cells(&d, p), cells(&mitred, p), "{p:?}");
    }
}

#[test]
fn guarded_seam_strokes_the_seam_as_a_join() {
    let f = fixture("guarded_seam");
    agrees(&f, &decided(f.name));
}

#[test]
fn the_guarded_outline_keeps_its_close_so_its_closing_join_is_stroked() {
    // `guarded_seam` undashed: the guarded outline, which the undashed stroke follows, keeps the
    // subpath's close, so (-4, -4) is a mitred join rather than two butt ends.
    let d = decide_at(&stroke(lines(&FAR), Joint::Miter, None, None), 1, None).unwrap();
    for s in [2, 3, 6, 7] {
        assert_eq!(
            admitted(&d, [3, 12], s),
            (255, 255, Cover::Yes as u8),
            "sample {s}"
        );
    }
}

#[test]
fn the_pinning_digest_is_sha256() {
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(
        sha256_hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
}
