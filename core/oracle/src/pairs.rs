//! A frame's depth at its samples (contract §5, §6, §7): every present fragment's band, its own
//! row adjusted on its own fill (`stroke_on_own_fill@2`) and its row on its record's plane
//! (`shared_plane_read@1`), and each present pair's admitted edges and label.

use std::collections::HashMap;

use crate::draw::Layer;
use crate::keys::{Band, Computation, Model, Unable, samples};
use crate::lines::LinesBound;
use crate::order::{Plate, Reads, Scene};
use crate::out::{BandRow, CoverRow, LABEL_UNABLE, PairRow};

/// A band row's status at a sample.
pub const STATUS_ABSENT: u8 = 0;
pub const STATUS_BOUNDED: u8 = 1;

fn status(b: &Result<Band, Unable>) -> u8 {
    match b {
        Ok(_) => STATUS_BOUNDED,
        Err(Unable::BandUnbounded) => 2,
        Err(Unable::CurveRecut) => 3,
        Err(Unable::PivotUnknown) => 4,
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

/// §7's `tied(a, e_a, b, e_b)` on two keys and their arithmetic bounds: equal, or both finite and
/// `|a − b| <= e_a + e_b`.
fn tied(a: &Band, b: &Band) -> bool {
    a.o == b.o || (a.o.is_finite() && b.o.is_finite() && (a.o - b.o).abs() <= a.r + b.r)
}

/// `stroke_on_own_fill@2` at one sample: the stroke's band where its fill covers too, `fill_either`
/// where the fill's cover is `Either`. It is keyed at the nearer of the two keys with that key's
/// bound, both bounds where they are `tied` (contract §5), and where the fill's cover is `Either`
/// the larger of that and its own. Its interval (§6, `band_input@2`) runs
/// from the nearer of the two near ends to the nearer of the two far ends, or, where the fill's
/// cover is `Either` and the core may keep the stroke's own key, to the stroke's own far end.
fn on_own_fill(
    own: Result<Band, Unable>,
    fill: Result<Band, Unable>,
    fill_either: bool,
) -> Result<Band, Unable> {
    let (o, f) = (own?, fill?);
    let taken = if f.o < o.o { f } else { o };
    let took = if tied(&o, &f) {
        crate::bound::max(o.r, f.r)
    } else {
        taken.r
    };
    // Where the fill's cover is `Either` the core may keep either key: the larger bound counts.
    let r = if fill_either {
        crate::bound::max(o.r, took)
    } else {
        took
    };
    let lo = crate::bound::least([o.lo(), f.lo()], f64::NEG_INFINITY);
    let hi = if fill_either {
        o.hi()
    } else {
        crate::bound::least([o.hi(), f.hi()], f64::INFINITY)
    };
    Ok(Band {
        o: taken.o,
        r,
        s_minus: crate::bound::max(taken.o - lo - 3.0 * r, 0.0),
        s_plus: crate::bound::max(hi - taken.o - 3.0 * r, 0.0),
    })
}

/// A fragment's band at each of a pixel's 16 samples.
type SampleBands = [Result<Band, Unable>; 16];

/// `stroke_on_own_fill@2` over a pixel: a stroke's bands `own` adjusted on its own fill's `fill`
/// at the samples both cover (`stroke` and `fill_cover` their `COVR` codes), and those samples;
/// `None` where they share none.
fn on_own_fill_pixel(
    own: &SampleBands,
    fill: &SampleBands,
    stroke: &[u8; 16],
    fill_cover: &[u8; 16],
) -> Option<(SampleBands, [bool; 16])> {
    let mut next = *own;
    let mut adjusted = [false; 16];
    for s in 0..16 {
        if stroke[s] == 0 || fill_cover[s] == 0 {
            continue;
        }
        next[s] = on_own_fill(own[s], fill[s], fill_cover[s] == 2);
        adjusted[s] = true;
    }
    if !adjusted.contains(&true) {
        return None;
    }
    // Its bound becomes the largest, over the samples, of the bound of the key it took there,
    // its own where the fill does not cover. Where any of those reads is not bounded
    // (`curve_recut` is per read, §6), neither is that largest, which every sample of the stroke
    // at the pixel takes: each is unable as that read is.
    if let Some(u) = (0..16)
        .filter(|&s| stroke[s] != 0)
        .find_map(|s| next[s].err())
    {
        for (s, b) in next.iter_mut().enumerate() {
            if stroke[s] != 0 {
                *b = Err(u);
            }
        }
        return Some((next, adjusted));
    }
    let r = (0..16)
        .filter(|&s| stroke[s] != 0)
        .filter_map(|s| next[s].ok().map(|b| b.r))
        .fold(0.0, crate::bound::max);
    for b in next.iter_mut().flatten() {
        b.r = crate::bound::max(b.r, r);
    }
    Some((next, adjusted))
}

/// The frame's bands and pairs over `covers` (sorted by `(y, x, fragment)`), writing each row's
/// own key and `R` into its `depth` and `band`. `offered` is the same fragments with the records
/// that paint nothing offered to the registry too (contract §5, `invisible_registry_offer`): each
/// pair's edges there are written beside its own.
pub fn frame(
    scene: &Scene,
    offered: &Scene,
    labelled: &[bool],
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
        pixel(
            scene,
            offered,
            labelled,
            &mut covers[i0..i1],
            &mut bands,
            &mut pairs,
        );
        i0 = i1;
    }
    (bands, pairs)
}

fn pixel(
    scene: &Scene,
    offered: &Scene,
    labelled: &[bool],
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
    // Where a stroke's own key is adjusted on its own fill, and, where the fill's cover is
    // `Either` there, its own key unadjusted: the core may keep it (§5, `stroke_on_own_fill@2`).
    let mut on_fill = vec![[false; 16]; rows.len()];
    let mut unadjusted: Vec<Option<([u8; 16], SampleBands)>> = vec![None; rows.len()];
    let row: Vec<Option<[Result<Band, Unable>; 16]>> = frags
        .iter()
        .map(|f| scene.frags[*f].row.as_ref().map(|m| m.pixel(px, py, true)))
        .collect();
    // `stroke_on_own_fill@2`: a stroke (never a background stroke) on its own fill.
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
        if let Some((next, adjusted)) =
            on_own_fill_pixel(&own[i], &own[j], &rows[i].cover, &rows[j].cover)
        {
            let either: [u8; 16] =
                std::array::from_fn(|s| (adjusted[s] && rows[j].cover[s] == 2) as u8);
            if either.contains(&1) {
                unadjusted[i] = Some((either, own[i]));
            }
            own[i] = next;
            on_fill[i] = adjusted;
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
        if let Some((either, before)) = &unadjusted[i] {
            bands.push(band_row(x, y, r.fragment, 2, either, before));
        }
    }
    // The `lines@1` shares, read once per fragment where a pair needs them.
    let pts = samples(px, py);
    // What each read computes: an own key adjusted on its fill is no one model's read.
    let own_at: Vec<[Option<Computation>; 16]> = frags
        .iter()
        .enumerate()
        .map(|(i, f)| {
            std::array::from_fn(|s| {
                (!on_fill[i][s])
                    .then(|| scene.frags[*f].model.computation_at(pts[s]))
                    .flatten()
            })
        })
        .collect();
    let row_at: Vec<[Option<Computation>; 16]> = frags
        .iter()
        .map(|f| {
            std::array::from_fn(|s| {
                scene.frags[*f]
                    .row
                    .as_ref()
                    .and_then(|m| m.computation_at(pts[s]))
            })
        })
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
    // A path's row against a fragment it shares an axis plane with, read once per pair.
    type AxisRow = Option<[Result<Band, Unable>; 16]>;
    let mut axis_rows: HashMap<(usize, usize), AxisRow> = HashMap::new();
    let mut axis_row = |i: usize, j: usize| -> AxisRow {
        *axis_rows.entry((i, j)).or_insert_with(|| {
            scene
                .axis_row(frags[i], frags[j])
                .map(|m| m.pixel(px, py, true))
        })
    };
    for s in 0..16 {
        let present: Vec<usize> = (0..rows.len()).filter(|&i| rows[i].cover[s] != 0).collect();
        for (n, &i) in present.iter().enumerate() {
            for &j in &present[n + 1..] {
                let ra = Reads {
                    own: own[i][s],
                    row: row[i].map(|r| r[s]),
                    axis_row: axis_row(i, j).map(|r| r[s]),
                    own_at: own_at[i][s].clone(),
                    row_at: row_at[i][s].clone(),
                };
                let rb = Reads {
                    own: own[j][s],
                    row: row[j].map(|r| r[s]),
                    axis_row: axis_row(j, i).map(|r| r[s]),
                    own_at: own_at[j][s].clone(),
                    row_at: row_at[j][s].clone(),
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
                let (with_invisible, offered_provenance) = if std::ptr::eq(scene, offered) {
                    (edges, provenance)
                } else {
                    match offered.pair(frags[i], frags[j], &ra, &rb, shares_ij) {
                        Ok(p) => (p.edges.bits(), p.provenance()),
                        Err(_) => (7, 0),
                    }
                };
                // A label's pair as the plate model orders it, where its keys establish that;
                // any other pair's sets are its own.
                let plate = if labelled[frags[i]] || labelled[frags[j]] {
                    scene
                        .plate_pair(frags[i], frags[j], &ra, &rb, shares_ij)
                        .unwrap_or(Plate::Undetermined)
                        .code()
                } else {
                    edges
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
                    plate,
                    offered_provenance,
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn band(o: f64, r: f64, s: f64) -> Result<Band, Unable> {
        Ok(Band {
            o,
            r,
            s_minus: s,
            s_plus: s,
        })
    }

    #[test]
    fn an_adjusted_stroke_takes_both_bounds_only_where_its_keys_are_tied() {
        // §5: the stroke takes the bound of the key it took, both where the keys tie by §7's
        // `tied`: `|a − b| <= e_a + e_b` on the arithmetic bounds, not where the intervals meet.
        // Keys 5e-4 apart with R 1e-6 and 1e-9: their intervals (S 1e-3) meet, the keys do not tie.
        let apart = on_own_fill(band(-1.0, 1e-6, 1e-3), band(-1.0005, 1e-9, 1e-3), false).unwrap();
        assert_eq!((apart.o, apart.r), (-1.0005, 1e-9));
        // 5e-7 apart: tied, both bounds count.
        let tied =
            on_own_fill(band(-1.0, 1e-6, 1e-3), band(-1.0000005, 1e-9, 1e-3), false).unwrap();
        assert_eq!((tied.o, tied.r), (-1.0000005, 1e-6));
        // Keys 0.02 apart with R 0.01 and 0.001 and no S: the intervals [−1.03, −0.97] and
        // [−1.023, −1.017] meet, the keys do not tie, and the nearer fill's 0.001 is taken.
        let met = on_own_fill(band(-1.0, 0.01, 0.0), band(-1.02, 0.001, 0.0), false).unwrap();
        assert_eq!((met.o, met.r), (-1.02, 0.001));
    }

    #[test]
    fn where_both_cover_an_adjusted_stroke_ends_at_the_nearer_far_end() {
        // §6 `band_input@2`: where both cover certainly, from the nearer of the two near ends to
        // the nearer of their far ends. The fill (−1.5) wholly nearer than the stroke's own key
        // (−1): the interval ends at the fill's far end, never reaching the stroke's own keys,
        // which the nearer of the two never takes.
        let b = on_own_fill(band(-1.0, 1e-9, 1e-4), band(-1.5, 1e-9, 1e-4), false).unwrap();
        let (lo, hi) = (b.lo(), b.hi());
        assert!((lo - (-1.5 - 1e-4 - 3e-9)).abs() < 1e-12, "{lo}");
        assert!((hi - (-1.5 + 1e-4 + 3e-9)).abs() < 1e-12, "{hi}");
    }

    #[test]
    fn where_the_fill_s_cover_is_either_an_adjusted_stroke_reaches_its_own_far_end() {
        // §6 `band_input@2`: where the fill's cover is `Either`, from the nearer near end to the
        // far end of the stroke's own interval: the core may keep the stroke's own key.
        let b = on_own_fill(band(-1.0, 1e-9, 1e-4), band(-1.5, 1e-9, 1e-4), true).unwrap();
        assert!((b.lo() - (-1.5 - 1e-4 - 3e-9)).abs() < 1e-12, "{}", b.lo());
        assert!(b.hi() >= -1.0 + 1e-4 + 3e-9 - 1e-12, "{}", b.hi());
    }

    #[test]
    fn where_the_fill_s_cover_is_either_an_adjusted_stroke_takes_the_larger_bound() {
        // §5 `stroke_on_own_fill@2`: where the fill's cover is `Either`, the stroke's bound is the
        // larger of its own and the one it takes where both cover, since the core may take either.
        // Disjoint keys, the fill nearer and taken, the stroke's own bound the larger.
        let b = on_own_fill(band(-1.0, 1e-6, 1e-4), band(-1.5, 1e-9, 1e-4), true).unwrap();
        assert_eq!((b.o, b.r), (-1.5, 1e-6));
        // Where both cover certainly, the taken key's bound alone.
        let b = on_own_fill(band(-1.0, 1e-6, 1e-4), band(-1.5, 1e-9, 1e-4), false).unwrap();
        assert_eq!((b.o, b.r), (-1.5, 1e-9));
    }

    #[test]
    fn an_adjusted_stroke_whose_bound_rests_on_an_unable_read_is_unable_at_the_pixel() {
        // §5 `stroke_on_own_fill@2`: the stroke's bound is the largest, over the samples, of the
        // bound of the key it took. Covering samples 0 to 2, its fill 1 and 2, its own read at 2
        // `curve_recut`: the key it took at 2 is not stated, nor its bound, so neither is the
        // pixel's, at any of its samples.
        let mut own = [band(-1.0, 1e-13, 0.0); 16];
        own[2] = Err(Unable::CurveRecut);
        let fill = [band(-0.9, 1e-3, 0.0); 16];
        let mut stroke = [0u8; 16];
        stroke[..3].fill(1);
        let mut cover = [0u8; 16];
        cover[1..3].fill(1);
        let (next, adjusted) = on_own_fill_pixel(&own, &fill, &stroke, &cover).unwrap();
        assert_eq!(&adjusted[..3], &[false, true, true]);
        for (s, b) in next.iter().enumerate().take(3) {
            assert_eq!(*b, Err(Unable::CurveRecut), "sample {s}");
        }
        // Stated at every sample, the pixel's bound is the largest taken: the stroke's own key at
        // 1 and 2 is nearer, its bound 1e-13; sample 0 takes the same.
        own[2] = band(-1.0, 1e-13, 0.0);
        let (next, _) = on_own_fill_pixel(&own, &fill, &stroke, &cover).unwrap();
        assert_eq!(next[0].map(|b| b.r), Ok(1e-13));
        // The fill nearer at 1: its bound counts at every sample.
        let mut fill = fill;
        fill[1] = band(-1.2, 1e-3, 0.0);
        let (next, _) = on_own_fill_pixel(&own, &fill, &stroke, &cover).unwrap();
        assert_eq!(next[0].map(|b| b.r), Ok(1e-3));
        // The own bound counts where the fill does not cover: unstated at 0, the largest is not
        // stated either, at any sample of the stroke there, the adjusted ones included.
        let mut mixed = own;
        mixed[0] = Err(Unable::CurveRecut);
        let (next, _) = on_own_fill_pixel(&mixed, &fill, &stroke, &cover).unwrap();
        for (s, b) in next.iter().enumerate().take(3) {
            assert_eq!(*b, Err(Unable::CurveRecut), "mixed, sample {s}");
        }
        // A stroke its fill covers at no sample keeps its own reads, each per read.
        let mut alone = own;
        alone[0] = Err(Unable::CurveRecut);
        assert_eq!(on_own_fill_pixel(&alone, &fill, &stroke, &[0; 16]), None);
    }

    #[test]
    fn an_adjusted_stroke_takes_its_most_severe_unable_class() {
        // §6: an unbounded read leaves the pixel's bound unbounded, whatever the other reads are;
        // the stroke's samples there take `band_unbounded`, the class a pair is first named by, not
        // the class of whichever unable read comes first in sample order. A `curve_recut` read at
        // sample 0 and an unbounded one at 1, the fill covering 2.
        let mut own = [band(-1.0, 1e-13, 0.0); 16];
        own[0] = Err(Unable::CurveRecut);
        own[1] = Err(Unable::BandUnbounded);
        let fill = [band(-0.9, 1e-3, 0.0); 16];
        let mut stroke = [0u8; 16];
        stroke[..3].fill(1);
        let mut cover = [0u8; 16];
        cover[2] = 1;
        let (next, _) = on_own_fill_pixel(&own, &fill, &stroke, &cover).unwrap();
        for (s, b) in next.iter().enumerate().take(3) {
            assert_eq!(*b, Err(Unable::BandUnbounded), "sample {s}");
        }
    }
}
