//! The proposed Frame IR record kind `composite`: its fixtures' bytes are pinned, and they read
//! back through the crate's reader. The writer here follows the layout in `composite.rs`; the
//! pinned bytes were written apart from it, from the layout's text.

use arachne_gpu::Focus;
use arachne_gpu::composite::{
    BLUR, Composite, CompositeMask, CompositeWindow, DEPTH, Entry, FOCUS, MASK, WINDOW,
    read_composite,
};

/// Every field present: flags 31.
const FULL: &str = "1f0000000000000440000000000000f4bf0000000000000840000000000000e03f03000000010203\
                    00000000000010c00000000000000240000000000000000000000000000010400000000000000240\
                    000000000000000000000000000010c000000000000002c000000000000000000000000000001040\
                    00000000000002c000000000000000000002000000000700000000000000010900000002000000\
                    01000000020b00000001000000000000000000e83f01";

/// No effect: flags 0 and one entry.
const MINIMAL: &str = "0001000000000300000000000000";

fn full() -> Composite {
    Composite {
        sigma: Some(2.5),
        depth: Some(-1.25),
        focus: Some(Focus {
            depth: 3.0,
            aperture: 0.5,
        }),
        window: Some(CompositeWindow {
            view: vec![1, 2, 3],
            corners: [
                [-4.0, 2.25, 0.0],
                [4.0, 2.25, 0.0],
                [-4.0, -2.25, 0.0],
                [4.0, -2.25, 0.0],
            ],
            resampling: 0,
        }),
        entries: vec![
            Entry {
                layer: 0,
                slot: 7,
                part: 0,
            },
            Entry {
                layer: 1,
                slot: 9,
                part: 2,
            },
        ],
        mask: Some(CompositeMask {
            entries: vec![Entry {
                layer: 2,
                slot: 11,
                part: 1,
            }],
            feather: 0.75,
            invert: true,
        }),
    }
}

fn minimal() -> Composite {
    Composite {
        sigma: None,
        depth: None,
        focus: None,
        window: None,
        entries: vec![Entry {
            layer: 0,
            slot: 3,
            part: 0,
        }],
        mask: None,
    }
}

fn write_entries(out: &mut Vec<u8>, entries: &[Entry]) {
    out.extend((entries.len() as u32).to_le_bytes());
    for entry in entries {
        out.push(entry.layer);
        out.extend(entry.slot.to_le_bytes());
        out.extend(entry.part.to_le_bytes());
    }
}

fn write(c: &Composite) -> Vec<u8> {
    let flag = |present: bool, flag: u8| if present { flag } else { 0 };
    let flags = flag(c.sigma.is_some(), BLUR)
        | flag(c.mask.is_some(), MASK)
        | flag(c.depth.is_some(), DEPTH)
        | flag(c.focus.is_some(), FOCUS)
        | flag(c.window.is_some(), WINDOW);
    let mut out = vec![flags];
    for value in [c.sigma, c.depth].into_iter().flatten() {
        out.extend(value.to_le_bytes());
    }
    if let Some(focus) = c.focus {
        out.extend(focus.depth.to_le_bytes());
        out.extend(focus.aperture.to_le_bytes());
    }
    if let Some(window) = &c.window {
        out.extend((window.view.len() as u32).to_le_bytes());
        out.extend(&window.view);
        for value in window.corners.iter().flatten() {
            out.extend(value.to_le_bytes());
        }
        out.push(window.resampling);
    }
    write_entries(&mut out, &c.entries);
    if let Some(mask) = &c.mask {
        write_entries(&mut out, &mask.entries);
        out.extend(mask.feather.to_le_bytes());
        out.push(u8::from(mask.invert));
    }
    out
}

fn bytes(hex: &str) -> Vec<u8> {
    let digits: Vec<u8> = hex.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    digits
        .chunks(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

#[test]
fn the_fixtures_bytes_are_pinned_and_read_back() {
    for (fixture, pinned) in [(full(), FULL), (minimal(), MINIMAL)] {
        let pinned = bytes(pinned);
        assert_eq!(write(&fixture), pinned);
        assert_eq!(read_composite(&pinned), Ok(fixture));
    }
    assert_eq!(bytes(FULL).len(), 181);
}

#[test]
fn bytes_a_newer_minor_appended_are_skipped() {
    let mut appended = bytes(FULL);
    appended.extend([9, 9, 9, 9, 9]);
    assert_eq!(read_composite(&appended), Ok(full()));
}

#[test]
fn a_cut_body_an_unknown_flag_or_a_bad_bool_is_unreadable() {
    let pinned = bytes(FULL);
    for end in 0..pinned.len() {
        assert!(read_composite(&pinned[..end]).is_err(), "cut at {end}");
    }
    let mut unknown = bytes(MINIMAL);
    unknown[0] = 32;
    assert!(read_composite(&unknown).is_err());
    let mut bad_bool = pinned.clone();
    *bad_bool.last_mut().unwrap() = 2;
    assert!(read_composite(&bad_bool).is_err());
    // A count past the body's end.
    let mut overcount = bytes(MINIMAL);
    overcount[1] = 200;
    assert!(read_composite(&overcount).is_err());
}
