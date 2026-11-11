//! XF 1.5's SOURCE fixtures (`core/tests/fixtures/xf15/`), read by the oracle alone: each file
//! hex-decoded under `STREAMS.md`'s rules, its decoded sha256 checked against `STREAMS.md`
//! before anything else is read, then read and applied by the oracle's own reading of `XF.md`
//! (`source.rs`, `stream.rs`), and what that reading makes of every delta diffed against
//! `expected.json` field by field. A disagreement is a finding, never settled here.

mod support;

use std::collections::BTreeMap;
use std::path::PathBuf;

use arachne_oracle::source::{Unread, read_file_bytes};
use arachne_oracle::stream::{Reason, State};
use support::{Json, decode_hex_text, parse_json, sha256_hex};

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/xf15")
}

/// `STREAMS.md`'s table: each file's decoded sha256 and its hex text's.
fn streams_table() -> BTreeMap<String, (String, String)> {
    let text = std::fs::read_to_string(dir().join("STREAMS.md")).expect("STREAMS.md");
    let mut out = BTreeMap::new();
    for line in text.lines() {
        let cells: Vec<&str> = line.split('|').map(str::trim).collect();
        if cells.len() == 5
            && let [name, decoded, hex] =
                [cells[1], cells[2], cells[3]].map(|c| c.trim_matches('`'))
            && name.ends_with(".arxf")
            && decoded.len() == 64
            && hex.len() == 64
        {
            out.insert(name.to_string(), (decoded.to_string(), hex.to_string()));
        }
    }
    out
}

/// What the oracle's reading makes of one delta, in `expected.json`'s terms.
fn reading(state: &mut State, w: &arachne_oracle::source::WireDelta) -> Json {
    let seq = Json::Num(w.seq as f64);
    if state.refused.is_some() {
        return Json::Obj(vec![("after_refusal".into(), Json::Bool(true))]);
    }
    let applied = state.apply_wire(w);
    match applied {
        Ok(()) => {
            let records = state
                .records
                .keys()
                .map(|slot| {
                    let meta = state.meta.get(slot).cloned().unwrap_or_default();
                    let source = meta
                        .source
                        .iter()
                        .map(|p| {
                            Json::Arr(
                                p.iter()
                                    .map(|(token, class)| {
                                        Json::Arr(vec![
                                            Json::Num(*token as f64),
                                            Json::Str(class.clone()),
                                        ])
                                    })
                                    .collect(),
                            )
                        })
                        .collect();
                    Json::Obj(vec![
                        ("slot".into(), Json::Num(*slot as f64)),
                        ("source".into(), Json::Arr(source)),
                        ("facing_group".into(), Json::Num(meta.facing_group as f64)),
                    ])
                })
                .collect();
            Json::Obj(vec![
                ("seq".into(), seq),
                ("minor".into(), Json::Num(w.version.1 as f64)),
                (
                    "read_under".into(),
                    Json::Str(if state.rules15 { "1.5" } else { "1.4" }.into()),
                ),
                ("records".into(), Json::Arr(records)),
            ])
        }
        Err(r) if r.reason == Reason::StreamRefused => {
            let (at, cause) = r.stream.expect("a stream refusal carries its delta");
            Json::Obj(vec![
                ("refused".into(), Json::Str(cause.name().into())),
                ("at_seq".into(), Json::Num(at as f64)),
            ])
        }
        Err(_) => {
            let (stage, error) = match &w.read {
                Err(Unread::Ordinary { error, .. }) => ("decode", error.clone()),
                // Apply's ordinary refusals carry the oracle's words, not the core's names; no
                // fixture has one.
                _ => ("apply", "apply".to_string()),
            };
            Json::Obj(vec![
                (
                    "refused".into(),
                    Json::Obj(vec![
                        ("stage".into(), Json::Str(stage.into())),
                        ("error".into(), Json::Str(error)),
                    ]),
                ),
                ("at_seq".into(), seq),
            ])
        }
    }
}

/// Every leaf of `expected` compared with the same place in `got`: `(fields, disagreements)`.
fn diff(path: &str, expected: &Json, got: Option<&Json>, out: &mut (usize, Vec<String>)) {
    match expected {
        Json::Obj(kv) => {
            for (k, v) in kv {
                diff(&format!("{path}.{k}"), v, got.and_then(|g| g.get(k)), out);
            }
            if let Some(Json::Obj(g)) = got {
                for (k, _) in g {
                    if expected.get(k).is_none() {
                        out.0 += 1;
                        out.1.push(format!("{path}.{k}: not expected, read"));
                    }
                }
            }
        }
        Json::Arr(a) => {
            let g = got.map(Json::arr).unwrap_or(&[]);
            out.0 += 1;
            if g.len() != a.len() {
                out.1
                    .push(format!("{path}: {} expected, {} read", a.len(), g.len()));
            }
            for (i, v) in a.iter().enumerate() {
                diff(&format!("{path}[{i}]"), v, g.get(i), out);
            }
        }
        leaf => {
            out.0 += 1;
            if got != Some(leaf) {
                out.1
                    .push(format!("{path}: expected {leaf:?}, read {got:?}"));
            }
        }
    }
}

#[test]
fn xf15_fixtures_read_as_xf_md_states() {
    let table = streams_table();
    let expected =
        parse_json(&std::fs::read_to_string(dir().join("expected.json")).expect("expected.json"))
            .expect("expected.json parses");
    assert_eq!(
        expected.get("schema").and_then(Json::str),
        Some("xf15-fixtures/1")
    );
    let files = expected.get("files").map(Json::arr).unwrap_or(&[]);
    assert!(!files.is_empty());
    assert_eq!(
        table.len(),
        files.len(),
        "STREAMS.md's table and expected.json's files"
    );
    let mut findings = Vec::new();
    let (mut fields, mut disagree) = (0, 0);
    for f in files {
        let name = f.get("name").and_then(Json::str).expect("a name");
        let text = std::fs::read(dir().join(format!("{name}.hex"))).expect("the hex text");
        let bytes = decode_hex_text(&text).unwrap_or_else(|e| panic!("{name}.hex: {e}"));
        let (want_decoded, want_hex) = table
            .get(name)
            .unwrap_or_else(|| panic!("{name}: not in STREAMS.md"));
        let decoded = sha256_hex(&bytes);
        assert_eq!(
            &decoded, want_decoded,
            "{name}: decoded sha256 against STREAMS.md"
        );
        assert_eq!(
            f.get("sha256").and_then(Json::str),
            Some(decoded.as_str()),
            "{name}: decoded sha256 against expected.json"
        );
        assert_eq!(
            &sha256_hex(&text),
            want_hex,
            "{name}.hex: sha256 against STREAMS.md"
        );
        println!("{name}: decoded sha256 ok");
        let (version, deltas) =
            read_file_bytes(&bytes).unwrap_or_else(|e| panic!("{name}: {}", e.0));
        let mut got_deltas = Vec::new();
        let mut state = State::new();
        for w in &deltas {
            got_deltas.push(reading(&mut state, w));
        }
        let got = Json::Obj(vec![
            ("name".into(), Json::Str(name.into())),
            ("sha256".into(), Json::Str(decoded.clone())),
            ("file_minor".into(), Json::Num(version.1 as f64)),
            ("deltas".into(), Json::Arr(got_deltas)),
        ]);
        let mut out = (0, Vec::new());
        diff(name, f, Some(&got), &mut out);
        println!(
            "{name}: fields: {}, agree {}, disagree {}",
            out.0,
            out.0 - out.1.len(),
            out.1.len()
        );
        fields += out.0;
        disagree += out.1.len();
        findings.extend(out.1);
    }
    println!(
        "xf15: {} files, fields: {fields}, agree {}, disagree {disagree}",
        files.len(),
        fields - disagree
    );
    assert!(
        findings.is_empty(),
        "disagreements:\n{}",
        findings.join("\n")
    );
}

/// End to end on each refused fixture: a valid delta, the refused one, then two valid ones (the
/// file's third delta and a copy of it at the next seq). The first frame is decided; the refused
/// delta's frame and both later ones are `truth_unable` under one `StreamRefused`, the refused
/// delta's seq and code, which the valid deltas after it never clear.
#[test]
fn xf15_fixtures_a_refused_stream_stays_refused() {
    use arachne_oracle::decide::decide_at_wire;
    use arachne_oracle::out::{STATUS_DECIDED, STATUS_UNABLE};
    use arachne_oracle::source::split_file;
    let expected =
        parse_json(&std::fs::read_to_string(dir().join("expected.json")).expect("expected.json"))
            .unwrap();
    let mut checked = 0;
    for f in expected.get("files").map(Json::arr).unwrap_or(&[]) {
        let name = f.get("name").and_then(Json::str).unwrap();
        let deltas = f.get("deltas").map(Json::arr).unwrap();
        let Some(code) = deltas
            .get(1)
            .and_then(|d| d.get("refused"))
            .and_then(Json::str)
        else {
            continue;
        };
        let bytes =
            decode_hex_text(&std::fs::read(dir().join(format!("{name}.hex"))).unwrap()).unwrap();
        let (version, items) = split_file(&bytes).unwrap();
        assert_eq!(items.len(), 3, "{name}");
        let mut fourth = items[2].to_vec();
        fourth[20..28].copy_from_slice(&3u64.to_le_bytes());
        let mut file = b"ARXF".to_vec();
        file.extend_from_slice(&version.0.to_le_bytes());
        file.extend_from_slice(&version.1.to_le_bytes());
        file.extend_from_slice(&4u32.to_le_bytes());
        for item in items.iter().copied().chain([&fourth[..]]) {
            file.extend_from_slice(&(item.len() as u32).to_le_bytes());
            file.extend_from_slice(item);
        }
        let (_, wire) = read_file_bytes(&file).unwrap();
        let first = decide_at_wire(&wire, 0, None).unwrap();
        assert_eq!(first.header.status, STATUS_DECIDED, "{name}: frame 0");
        for seq in 1..4 {
            let d = decide_at_wire(&wire, seq, None).unwrap();
            assert_eq!(d.header.status, STATUS_UNABLE, "{name}: frame {seq}");
            assert_eq!(d.refused.len(), 1, "{name}: frame {seq}");
            let r = &d.refused[0];
            assert_eq!(
                r.reason,
                Reason::StreamRefused.code(),
                "{name}: frame {seq}"
            );
            assert_eq!(r.at_seq, 1, "{name}: frame {seq}");
            let cause = arachne_oracle::stream::StreamCause::from_code(r.cause).map(|c| c.name());
            assert_eq!(cause, Some(code), "{name}: frame {seq}");
        }
        checked += 1;
    }
    assert_eq!(checked, 11, "the refused fixtures, 08 to 18");
}
