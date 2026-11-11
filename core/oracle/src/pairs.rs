//! A frame's depth at its samples (contract §5, §6, §7): every present fragment's band, its own
//! row adjusted on its own fill (`stroke_on_own_fill@1`) and its row on its record's plane
//! (`shared_plane_read@1`), and each present pair's admitted edges and label.

use crate::draw::Layer;
use crate::keys::{Band, Model, Unable, samples};
use crate::lines::LinesBound;
use crate::order::{Reads, Scene};
use crate::out::{BandRow, CoverRow, LABEL_UNABLE, PairRow};

/// A band row's status at a sample.
pub const STATUS_ABSENT: u8 = 0;
pub const STATUS_BOUNDED: u8 = 1;

fn status(b: &Result<Band, Unable>) -> u8 {
    match b {
        Ok(_) => STATUS_BOUNDED,
        Err(Unable::BandUnbounded) => 2,
        Err(Unable::CurveRecut) => 3,
    }
}

fn band_row(
    x: u32,
    y: u32,
    fragment: u32,
    which: u8,
    cover: &[u8; 16],
    bands: &[Result<Band, Unable>; 16],
) -> BandRow {
    let mut row = BandRow {
        x,
        y,
        fragment,
        which,
        o: [f64::NAN; 16],
        r: [f64::NAN; 16],
        s_minus: [f64::NAN; 16],
        s_plus: [f64::NAN; 16],
        status: [STATUS_ABSENT; 16],
    };
    for s in 0..16 {
        if cover[s] == 0 {
            continue;
        }
        row.status[s] = status(&bands[s]);
        if let Ok(b) = bands[s] {
            row.o[s] = b.o;
            row.r[s] = b.r;
            row.s_minus[s] = b.s_minus;
            row.s_plus[s] = b.s_plus;
        }
    }
    row
}

/// The `lines@1` share terms of a stroke whose model reads a centreline.
pub fn lines_of(model: &Model) -> Option<LinesBound> {
    match model {
        Model::Curve {
            line,
            h,
            eps,
            sigma,
            joints,
            ..
        } => Some(LinesBound::of(line, joints, *h, *eps, *sigma)),
        _ => None,
    }
}

fn meets(a: &Band, b: &Band) -> bool {
    a.lo() <= b.hi() && b.lo() <= a.hi()
}

/// `stroke_on_own_fill@1` at one sample: the stroke's band where its fill covers too.
fn on_own_fill(
    own: Result<Band, Unable>,
    fill: Result<Band, Unable>,
    both_certain: bool,
) -> Result<Band, Unable> {
    let (o, f) = (own?, fill?);
    let taken = if meets(&o, &f) {
        o.hull(&f)
    } else if f.o < o.o {
        f
    } else {
        o
    };
    // Where either coverage is `Either`, the stroke may keep its own key.
    Ok(if both_certain { taken } else { o.hull(&taken) })
}

/// The frame's bands and pairs over `covers` (sorted by `(y, x, fragment)`), writing each row's
/// own key and `R` into its `depth` and `band`. `offered` is the same fragments with the records
/// that paint nothing offered to the registry too (contract §5, `invisible_registry_offer`): each
/// pair's edges there are written beside its own.
pub fn frame(
    scene: &Scene,
    offered: &Scene,
    covers: &mut [CoverRow],
) -> (Vec<BandRow>, Vec<PairRow>) {
    let mut bands = Vec::new();
    let mut pairs = Vec::new();
    let mut i0 = 0;
    while i0 < covers.len() {
        let (x, y) = (covers[i0].x, covers[i0].y);
        let mut i1 = i0;
        while i1 < covers.len() && covers[i1].x == x && covers[i1].y == y {
            i1 += 1;
        }
        pixel(scene, offered, &mut covers[i0..i1], &mut bands, &mut pairs);
        i0 = i1;
    }
    (bands, pairs)
}

fn pixel(
    scene: &Scene,
    offered: &Scene,
    rows: &mut [CoverRow],
    bands: &mut Vec<BandRow>,
    pairs: &mut Vec<PairRow>,
) {
    let (x, y) = (rows[0].x, rows[0].y);
    let (px, py) = (x as i64, y as i64);
    let frags: Vec<usize> = rows.iter().map(|r| r.fragment as usize).collect();
    let mut own: Vec<[Result<Band, Unable>; 16]> = frags
        .iter()
        .map(|f| scene.frags[*f].model.pixel(px, py, false))
        .collect();
    let row: Vec<Option<[Result<Band, Unable>; 16]>> = frags
        .iter()
        .map(|f| scene.frags[*f].row.as_ref().map(|m| m.pixel(px, py, true)))
        .collect();
    // `stroke_on_own_fill@1`: a stroke (never a background stroke) on its own fill.
    for i in 0..rows.len() {
        let fi = &scene.frags[frags[i]];
        if fi.layer != Layer::Stroke {
            continue;
        }
        let Some(j) = (0..rows.len()).find(|&j| {
            let fj = &scene.frags[frags[j]];
            fj.layer == Layer::Fill
                && fj.group == fi.group
                && fj.entry == fi.entry
                && fj.sub / 3 == fi.sub / 3
                && fj.sub <= fi.sub
        }) else {
            continue;
        };
        let mut adjusted = false;
        let mut next = own[i];
        for s in 0..16 {
            let (cs, cf) = (rows[i].cover[s], rows[j].cover[s]);
            if cs == 0 || cf == 0 {
                continue;
            }
            next[s] = on_own_fill(own[i][s], own[j][s], cs == 1 && cf == 1);
            adjusted = true;
        }
        if adjusted {
            // Its bound becomes the largest, over the samples, of the bound it took.
            let r = next
                .iter()
                .zip(&rows[i].cover)
                .filter(|(_, c)| **c != 0)
                .filter_map(|(b, _)| b.ok().map(|b| b.r))
                .fold(0.0, f64::max);
            for b in next.iter_mut().flatten() {
                b.r = b.r.max(r);
            }
            own[i] = next;
        }
    }
    for (i, r) in rows.iter_mut().enumerate() {
        for (s, band) in own[i].iter().enumerate() {
            let (d, e) = match (r.cover[s], band) {
                (0, _) | (_, Err(_)) => (f64::NAN, f64::NAN),
                (_, Ok(b)) => (b.o, b.r),
            };
            r.depth[s] = d;
            r.band[s] = e;
        }
        bands.push(band_row(x, y, r.fragment, 0, &r.cover, &own[i]));
        if let Some(rw) = &row[i] {
            bands.push(band_row(x, y, r.fragment, 1, &r.cover, rw));
        }
    }
    // The `lines@1` shares, read once per fragment where a pair needs them.
    let pts = samples(px, py);
    let at_sample: Vec<[bool; 16]> = frags
        .iter()
        .map(|f| std::array::from_fn(|s| scene.frags[*f].model.read_at(pts[s])))
        .collect();
    let mut shares: Vec<Option<f64>> = vec![None; rows.len()];
    let mut share = |i: usize| -> f64 {
        *shares[i].get_or_insert_with(|| {
            let f = &scene.frags[frags[i]];
            match (&f.lines, &f.model) {
                (Some(lb), Model::Curve { line, .. }) => lb.share(line, px, py, &pts),
                _ => f64::INFINITY,
            }
        })
    };
    for s in 0..16 {
        let present: Vec<usize> = (0..rows.len()).filter(|&i| rows[i].cover[s] != 0).collect();
        for (n, &i) in present.iter().enumerate() {
            for &j in &present[n + 1..] {
                let ra = Reads {
                    own: own[i][s],
                    row: row[i].map(|r| r[s]),
                    at_sample: at_sample[i][s],
                };
                let rb = Reads {
                    own: own[j][s],
                    row: row[j].map(|r| r[s]),
                    at_sample: at_sample[j][s],
                };
                let lines_pair = scene.lines_eligible(frags[i], frags[j]);
                let shares_ij = if lines_pair {
                    (share(i), share(j))
                } else {
                    (f64::INFINITY, f64::INFINITY)
                };
                let (edges, label, provenance) =
                    match scene.pair(frags[i], frags[j], &ra, &rb, shares_ij) {
                        Ok(p) => (p.edges.bits(), p.label.code(), p.provenance()),
                        Err(_) => (7, LABEL_UNABLE, 0),
                    };
                // With nothing more offered, the sets are the same ones.
                let with_invisible = if std::ptr::eq(scene, offered) {
                    edges
                } else {
                    match offered.pair(frags[i], frags[j], &ra, &rb, shares_ij) {
                        Ok(p) => p.edges.bits(),
                        Err(_) => 7,
                    }
                };
                pairs.push(PairRow {
                    x,
                    y,
                    a: rows[i].fragment,
                    b: rows[j].fragment,
                    sample: s as u8,
                    edges,
                    label,
                    provenance,
                    offered: with_invisible,
                });
            }
        }
    }
}
