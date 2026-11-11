//! The contract's refusal witnesses (§12, W28 "stream_refused", "Refusals, by code",
//! "Precedence"): four deltas over W27's scene, the second the one refused. A source code refuses
//! the stream from that delta on, the valid deltas after it clearing nothing; an ordinary refusal
//! refuses its own frame alone. Each stream is built here, pinned by `scene.sha256`, and its
//! frames' expected outcomes are `expected.json` (`ORACLE_BLESS_FIXTURES=1` rewrites both).

mod common;
#[path = "common/scenes.rs"]
mod scenes;

use std::path::{Path, PathBuf};

use arachne_core::xf::{Delta, Layer, OrderEntry, Record, Time, wire};
use arachne_oracle::decide::decide_at_wire;
use arachne_oracle::source;
use arachne_oracle::stream::{Reason, StreamCause};
use common::sha256_hex;
use scenes::*;

const POLYGON: &str = "arachne.mobject.geometry.polygram.Polygon";

/// What delta 2 is: its in-memory changes and its section's bytes.
struct Second {
    set: Vec<Record>,
    remove: Vec<u32>,
    order: Option<Vec<u32>>,
    section: Vec<u8>,
    /// The order's first entry's layer byte replaced by an unknown code.
    bad_layer: bool,
}

fn second(section: Vec<u8>) -> Second {
    Second {
        set: Vec::new(),
        remove: Vec::new(),
        order: None,
        section,
        bad_layer: false,
    }
}

/// How a frame is expected to come out.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Outcome {
    Judged,
    Stream(StreamCause),
    Invalid,
}

fn bare(seq: u64) -> Delta {
    let mut d = Delta::new(
        Time {
            num: seq as i64,
            den: 1,
        },
        seq,
    );
    d.version = (1, 4);
    d
}

fn order(slots: &[u32]) -> Vec<OrderEntry> {
    slots
        .iter()
        .map(|&slot| OrderEntry {
            layer: Layer::Moving,
            slot,
            part: 0,
        })
        .collect()
}

/// W27's scene in delta 1 (slot 0 `G`, slot 1 `L`, each with a row), `two` as delta 2, then two
/// deltas that each move `L`.
fn stream(two: Second) -> Vec<u8> {
    let mut b = Build::new(true);
    b.fill(rect([-3.0, 3.0], [-3.0, 3.0], 2.0), OPAQUE_BLUE);
    b.label([0.0; 3]);
    let rows = vec![row(0, 0, &[&[(1, POLYGON)]]), row(1, 1, &[&[(2, POLYGON)]])];
    let one = delta15(&b.delta(1, view_c()), &section(&rows, None));
    let mut d2 = bare(2);
    d2.set = two.set;
    d2.remove = two.remove;
    d2.order = two.order.map(|o| order(&o));
    let mut bytes2 = delta15(&d2, &[]);
    if two.bad_layer {
        let n = d2.order.as_ref().unwrap().len();
        let at = bytes2.len() - 9 * n;
        bytes2[at] = 9;
    }
    bytes2.extend(two.section);
    let mut out = vec![one, bytes2];
    for seq in [3, 4] {
        let mut l = b.records[1].clone();
        l.geometry.place_a[0][3] = 0.25 * (seq - 2) as f64;
        let mut d = bare(seq);
        d.set = vec![l];
        out.push(wire::encode_delta(&d));
    }
    file15(&out)
}

/// A record for a slot no delta has set.
fn new_record(slot: u32) -> Record {
    let mut b = Build::new(true);
    b.fill(rect([-1.0, 1.0], [-1.0, 1.0], 0.0), OPAQUE_RED);
    let mut r = b.records[0].clone();
    r.slot = slot;
    r.object = 100 + slot;
    // Its shape is delta 1's `G`'s, held.
    r.geometry.shape_a = Build::new(true)
        .fill(rect([-3.0, 3.0], [-3.0, 3.0], 2.0), OPAQUE_BLUE)
        .geometry
        .shape_a;
    r
}

fn one_path(slot: u32, group: u32) -> (u32, u32, Vec<Vec<(u32, u32)>>) {
    (slot, group, vec![vec![(1, 0)]])
}

fn cases() -> Vec<(&'static str, Second, Outcome)> {
    use StreamCause::*;
    let p = &[POLYGON][..];
    let mut out = vec![
        (
            "stream_refused",
            second(raw_section(p, &[one_path(9, 0)])),
            Outcome::Stream(SlotNotHeld),
        ),
        (
            "refused_class_named_twice",
            second(raw_section(&[POLYGON, POLYGON], &[one_path(0, 0)])),
            Outcome::Stream(ClassNamedTwice),
        ),
        (
            "refused_slot_named_twice",
            second(raw_section(p, &[one_path(0, 0), one_path(0, 0)])),
            Outcome::Stream(SlotNamedTwice),
        ),
        (
            "refused_slot_removed",
            Second {
                remove: vec![0],
                order: Some(vec![1]),
                ..second(raw_section(p, &[one_path(0, 0)]))
            },
            Outcome::Stream(SlotRemoved),
        ),
        (
            "refused_set_and_removed",
            Second {
                set: vec![new_record(5)],
                remove: vec![5],
                ..second(raw_section(p, &[one_path(5, 0)]))
            },
            Outcome::Stream(SlotRemoved),
        ),
        (
            "refused_row_empty",
            second(raw_section(p, &[(0, 0, vec![])])),
            Outcome::Stream(RowEmpty),
        ),
        (
            "refused_set_row_empty",
            Second {
                set: vec![new_record(5)],
                ..second(raw_section(p, &[(5, 0, vec![])]))
            },
            Outcome::Stream(RowEmpty),
        ),
        (
            "refused_path_empty",
            second(raw_section(p, &[(0, 0, vec![vec![]])])),
            Outcome::Stream(PathEmpty),
        ),
        (
            "refused_class_out_of_range",
            second(raw_section(p, &[(0, 0, vec![vec![(1, 1)]])])),
            Outcome::Stream(ClassOutOfRange),
        ),
        (
            "refused_class_before_rows",
            Second {
                remove: vec![0],
                order: Some(vec![1]),
                ..second(raw_section(&[POLYGON, POLYGON], &[one_path(0, 0)]))
            },
            Outcome::Stream(ClassNamedTwice),
        ),
        (
            "refused_first_offending_row",
            Second {
                remove: vec![0],
                order: Some(vec![1]),
                ..second(raw_section(p, &[(0, 0, vec![]), (1, 1, vec![vec![]])]))
            },
            Outcome::Stream(SlotRemoved),
        ),
        (
            "refused_twice_before_removed",
            Second {
                remove: vec![1],
                order: Some(vec![0]),
                ..second(raw_section(
                    p,
                    &[one_path(0, 0), one_path(0, 0), one_path(1, 1)],
                ))
            },
            Outcome::Stream(SlotNamedTwice),
        ),
        (
            "refused_reading_before_apply",
            second(raw_section(p, &[one_path(9, 0), (0, 0, vec![vec![]])])),
            Outcome::Stream(PathEmpty),
        ),
        (
            "refused_apply_before_slot_not_held",
            Second {
                order: Some(vec![0, 1, 7]),
                ..second(raw_section(p, &[one_path(9, 0)]))
            },
            Outcome::Invalid,
        ),
    ];
    // `t` set again naming a shape the delta neither uploads nor finds held.
    let mut t = Build::new(true);
    t.label([0.0; 3]);
    let mut lost = t.records[0].clone();
    lost.slot = 1;
    lost.object = 101;
    lost.geometry.shape_a = Build::new(true)
        .fill(rect([-9.0, 9.0], [-9.0, 9.0], 0.0), OPAQUE_RED)
        .geometry
        .shape_a;
    out.push((
        "refused_section_before_apply",
        Second {
            set: vec![lost],
            remove: vec![0],
            order: Some(vec![1]),
            ..second(raw_section(p, &[one_path(0, 0)]))
        },
        Outcome::Stream(SlotRemoved),
    ));
    out.push((
        "refused_decode_before_section",
        Second {
            order: Some(vec![0, 1]),
            bad_layer: true,
            ..second(raw_section(&[POLYGON, POLYGON], &[one_path(0, 0)]))
        },
        Outcome::Invalid,
    ));
    out
}

fn outcome_word(o: Outcome) -> String {
    match o {
        Outcome::Judged => "judged".into(),
        Outcome::Stream(c) => format!("stream_refused at_seq 2 cause {}", c.name()),
        Outcome::Invalid => "invalid_delta".into(),
    }
}

fn dir(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

#[test]
fn a_refused_delta_refuses_what_its_code_says_and_valid_deltas_after_it_clear_nothing() {
    let bless = std::env::var_os("ORACLE_BLESS_FIXTURES").is_some();
    let mut wrong = Vec::new();
    for (name, two, outcome) in cases() {
        let bytes = stream(two);
        let (_, deltas) = source::read_file_bytes(&bytes).unwrap();
        let mut frames = Vec::new();
        for seq in 1..=4u64 {
            let want = match (seq, outcome) {
                (1, _) => Outcome::Judged,
                (2, o) => o,
                (_, Outcome::Invalid) => Outcome::Judged,
                (_, o) => o,
            };
            let d = decide_at_wire(&deltas, seq, None).unwrap();
            let got = if d.header.status == 0 {
                Outcome::Judged
            } else if let Some(r) = d
                .refused
                .iter()
                .find(|r| r.reason == Reason::StreamRefused.code())
            {
                match StreamCause::from_code(r.cause) {
                    Some(c) if r.at_seq == 2 => Outcome::Stream(c),
                    _ => Outcome::Invalid,
                }
            } else if d
                .refused
                .iter()
                .all(|r| r.reason == Reason::InvalidDelta.code())
            {
                Outcome::Invalid
            } else {
                wrong.push(format!("{name} frame {seq}: {:?}", d.refused));
                continue;
            };
            if got != want {
                wrong.push(format!(
                    "{name} frame {seq}: {} not {} ({:?})",
                    outcome_word(got),
                    outcome_word(want),
                    d.refused.iter().map(|r| &r.detail).collect::<Vec<_>>()
                ));
            }
            frames.push(format!(
                "    {{\"seq\": {seq}, \"outcome\": \"{}\"}}",
                outcome_word(want)
            ));
        }
        let digest = format!("{}\n", sha256_hex(&bytes));
        let expected = format!(
            "{{\n  \"fixture\": \"{name}\",\n  \"kind\": \"refusal\",\n  \"frames\": [\n{}\n  ]\n}}\n",
            frames.join(",\n")
        );
        let path = dir(name);
        if bless {
            std::fs::create_dir_all(&path).unwrap();
            std::fs::write(path.join("scene.sha256"), &digest).unwrap();
            std::fs::write(path.join("expected.json"), &expected).unwrap();
        }
        let read = |f: &str| std::fs::read_to_string(path.join(f)).ok();
        if read("scene.sha256").as_deref() != Some(digest.as_str()) {
            wrong.push(format!("{name}: scene.sha256 does not pin the scene"));
        }
        if read("expected.json").as_deref() != Some(expected.as_str()) {
            wrong.push(format!("{name}: expected.json differs"));
        }
    }

    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
